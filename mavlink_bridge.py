"""MAVLink bridge for SITL and real multicopters.

Design goals:
- one adapter for PX4/ArduPilot compatible MAVLink endpoints;
- telemetry via GLOBAL_POSITION_INT / HEARTBEAT / SYS_STATUS;
- horizontal velocity commands via SET_POSITION_TARGET_LOCAL_NED;
- explicit safety interlock for real hardware;
- background 10 Hz setpoint stream (important for PX4 Offboard mode).

Real-hardware commands are disabled unless BOTH:
1) mode='REAL';
2) environment SWARM_ENABLE_REAL=1;
3) the UI passes operator_confirmed=True.

This is production-oriented integration code, not a certified flight-control system.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import os
import threading
import time
from typing import Optional

try:
    from pymavlink import mavutil
except Exception:  # optional dependency until real/SITL tab is used
    mavutil = None


@dataclass
class VehicleTelemetry:
    name: str
    endpoint: str
    system_id: int = 0
    component_id: int = 0
    autopilot: str = "UNKNOWN"
    connected: bool = False
    armed: bool = False
    mode: str = ""
    lat_deg: float | None = None
    lon_deg: float | None = None
    relative_alt_m: float | None = None
    battery_pct: float | None = None
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    last_heartbeat_s: float = 0.0
    last_position_s: float = 0.0
    status: str = "DISCONNECTED"


class MavlinkVehicle:
    def __init__(self,name:str,endpoint:str,baud:int=57600):
        self.name=name; self.endpoint=endpoint; self.baud=baud
        self.conn=None
        self.telemetry=VehicleTelemetry(name,endpoint)
        self.desired_velocity=(0.0,0.0,0.0)  # N,E,D m/s
        self.stream_enabled=False
        self.lock=threading.RLock()

    def connect(self,timeout=8.0):
        if mavutil is None:
            raise RuntimeError("pymavlink не установлен. Выполните: pip install pymavlink")
        kwargs={"source_system":250}
        if self.endpoint.lower().startswith("serial:"):
            # serial:COM4:57600 or serial:/dev/ttyACM0:115200
            parts=self.endpoint.split(":")
            device=parts[1]; baud=int(parts[2]) if len(parts)>2 else self.baud
            self.conn=mavutil.mavlink_connection(device,baud=baud,**kwargs)
        else:
            self.conn=mavutil.mavlink_connection(self.endpoint,**kwargs)
        hb=self.conn.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise TimeoutError(f"{self.name}: heartbeat не получен за {timeout} с")
        self.telemetry.system_id=int(self.conn.target_system)
        self.telemetry.component_id=int(self.conn.target_component)
        self.telemetry.connected=True
        self.telemetry.last_heartbeat_s=time.time()
        self.telemetry.autopilot=self._autopilot_name(getattr(hb,"autopilot",0))
        self.telemetry.status="CONNECTED"
        self._request_streams()
        return self.telemetry

    def _autopilot_name(self,value:int):
        if mavutil is None: return str(value)
        m=mavutil.mavlink
        if value==getattr(m,"MAV_AUTOPILOT_PX4",12): return "PX4"
        if value==getattr(m,"MAV_AUTOPILOT_ARDUPILOTMEGA",3): return "ARDUPILOT"
        return f"AUTOPILOT_{value}"

    def _request_streams(self):
        if not self.conn: return
        try:
            self.conn.mav.command_long_send(self.conn.target_system,self.conn.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,0,
                mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,100000,0,0,0,0,0)
            self.conn.mav.command_long_send(self.conn.target_system,self.conn.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,0,
                mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS,500000,0,0,0,0,0)
        except Exception:
            pass

    def poll(self,max_messages=30):
        if not self.conn: return
        now=time.time()
        for _ in range(max_messages):
            msg=self.conn.recv_match(blocking=False)
            if msg is None: break
            typ=msg.get_type()
            if typ=="BAD_DATA": continue
            if typ=="HEARTBEAT":
                self.telemetry.last_heartbeat_s=now
                base_mode=int(getattr(msg,"base_mode",0))
                self.telemetry.armed=bool(base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                try: self.telemetry.mode=mavutil.mode_string_v10(msg)
                except Exception: pass
            elif typ=="GLOBAL_POSITION_INT":
                self.telemetry.lat_deg=float(msg.lat)/1e7; self.telemetry.lon_deg=float(msg.lon)/1e7
                self.telemetry.relative_alt_m=float(msg.relative_alt)/1000.0
                self.telemetry.vx_mps=float(msg.vx)/100.0; self.telemetry.vy_mps=float(msg.vy)/100.0
                self.telemetry.last_position_s=now
            elif typ=="SYS_STATUS":
                rem=getattr(msg,"battery_remaining",-1)
                if rem is not None and rem>=0: self.telemetry.battery_pct=float(rem)
        age=now-self.telemetry.last_heartbeat_s
        self.telemetry.connected=age<3.0
        if age>=3.0: self.telemetry.status="HEARTBEAT LOST"

    def _command_long(self,command,*params):
        if not self.conn: raise RuntimeError("Нет соединения")
        p=list(params)+[0.0]*7; p=p[:7]
        self.conn.mav.command_long_send(self.conn.target_system,self.conn.target_component,command,0,*p)

    def set_mode(self,mode:str):
        if not self.conn: raise RuntimeError("Нет соединения")
        mapping=self.conn.mode_mapping() or {}
        if mode not in mapping:
            # aliases across PX4/ArduPilot
            aliases={"HOLD":["LOITER","POSCTL","BRAKE"],"GUIDED":["GUIDED"],"OFFBOARD":["OFFBOARD"]}
            for candidate in aliases.get(mode,[mode]):
                if candidate in mapping:
                    mode=candidate; break
            else:
                raise RuntimeError(f"Режим {mode} отсутствует в mode_mapping автопилота")
        self.conn.set_mode(mapping[mode])

    def arm(self):
        self._command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,1.0)

    def disarm(self):
        self._command_long(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,0.0)

    def takeoff(self,alt_m:float):
        self._command_long(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,0,0,0,0,0,0,float(alt_m))

    def land(self):
        self._command_long(mavutil.mavlink.MAV_CMD_NAV_LAND,0,0,0,0,0,0,0)

    def send_velocity_ned(self,vn:float,ve:float,vd:float=0.0,yaw_rate:float=0.0):
        if not self.conn: return
        # Ignore position, acceleration, yaw; use vx/vy/vz and yaw_rate.
        ignore_pos=(1<<0)|(1<<1)|(1<<2)
        ignore_acc=(1<<6)|(1<<7)|(1<<8)
        force=(1<<9)
        ignore_yaw=(1<<10)
        type_mask=ignore_pos|ignore_acc|force|ignore_yaw
        self.conn.mav.set_position_target_local_ned_send(
            int((time.time()*1000)%2**32),self.conn.target_system,self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,type_mask,
            0,0,0,float(vn),float(ve),float(vd),0,0,0,0,float(yaw_rate))

    def close(self):
        try:
            if self.conn: self.conn.close()
        finally:
            self.conn=None; self.telemetry.connected=False; self.telemetry.status="DISCONNECTED"


class MavlinkFleetManager:
    def __init__(self,rate_hz:float=10.0):
        self.vehicles:list[MavlinkVehicle]=[]
        self.rate_hz=max(3.0,float(rate_hz))
        self.mode="SITL"
        self.operator_confirmed=False
        self.running=False
        self.thread:Optional[threading.Thread]=None
        self.lock=threading.RLock()
        self.origin_lat=None; self.origin_lon=None
        self.workspace_offset=(40.0,30.0)
        self.control_active=False
        self.max_speed_mps=4.0
        self.last_error=""

    @property
    def real_enabled(self):
        return os.environ.get("SWARM_ENABLE_REAL","0")=="1"

    def _write_allowed(self):
        if self.mode=="SITL": return True
        return self.mode=="REAL" and self.real_enabled and self.operator_confirmed

    def connect(self,entries:list[tuple[str,str]],mode="SITL",operator_confirmed=False,timeout=6.0):
        self.disconnect()
        self.mode=mode; self.operator_confirmed=bool(operator_confirmed)
        if mode=="REAL" and not self._write_allowed():
            # Reading telemetry is allowed; flight commands remain blocked.
            pass
        connected=[]
        for name,endpoint in entries:
            v=MavlinkVehicle(name,endpoint)
            v.connect(timeout=timeout); connected.append(v)
        self.vehicles=connected
        self.running=True
        self.thread=threading.Thread(target=self._loop,name="mavlink-fleet-loop",daemon=True); self.thread.start()
        return self.telemetry_rows()

    def _loop(self):
        period=1.0/self.rate_hz
        while self.running:
            t0=time.perf_counter()
            try:
                for v in list(self.vehicles):
                    v.poll()
                    if self.control_active and self._write_allowed() and v.telemetry.connected:
                        vn,ve,vd=v.desired_velocity
                        speed=math.hypot(vn,ve)
                        if speed>self.max_speed_mps:
                            s=self.max_speed_mps/(speed+1e-9); vn*=s; ve*=s
                        v.send_velocity_ned(vn,ve,vd)
                self._update_origin()
            except Exception as exc:
                self.last_error=str(exc)
            elapsed=time.perf_counter()-t0
            time.sleep(max(0.0,period-elapsed))

    def _update_origin(self):
        if self.origin_lat is not None: return
        for v in self.vehicles:
            t=v.telemetry
            if t.lat_deg is not None and t.lon_deg is not None:
                self.origin_lat=t.lat_deg; self.origin_lon=t.lon_deg; return

    def local_xy(self,t:VehicleTelemetry):
        if self.origin_lat is None or t.lat_deg is None or t.lon_deg is None: return None
        R=6378137.0
        lat0=math.radians(self.origin_lat)
        north=math.radians(t.lat_deg-self.origin_lat)*R
        east=math.radians(t.lon_deg-self.origin_lon)*R*math.cos(lat0)
        return east+self.workspace_offset[0], north+self.workspace_offset[1]

    def telemetry_rows(self):
        rows=[]
        for v in self.vehicles:
            t=v.telemetry; xy=self.local_xy(t)
            rows.append({"БПЛА":t.name,"Endpoint":t.endpoint,"SysID":t.system_id,"Автопилот":t.autopilot,"Связь":t.connected,"ARM":t.armed,"Режим":t.mode,"X, м":None if xy is None else round(xy[0],2),"Y, м":None if xy is None else round(xy[1],2),"Высота, м":t.relative_alt_m,"Батарея, %":t.battery_pct,"Статус":t.status})
        return rows

    def set_workspace_offset(self,x:float,y:float): self.workspace_offset=(float(x),float(y))

    def set_velocity_xy(self,commands:dict[str,tuple[float,float]],max_speed=None):
        if max_speed is not None: self.max_speed_mps=float(max_speed)
        for v in self.vehicles:
            vx,vy=commands.get(v.name,(0.0,0.0))
            # App coordinates: X=East, Y=North. MAVLink NED: vx=North, vy=East.
            v.desired_velocity=(float(vy),float(vx),0.0)

    def prime_offboard(self,seconds=1.2):
        if not self._write_allowed(): raise PermissionError(self.interlock_message())
        self.control_active=True
        for v in self.vehicles: v.desired_velocity=(0.0,0.0,0.0)
        time.sleep(seconds)

    def arm_all(self):
        if not self._write_allowed(): raise PermissionError(self.interlock_message())
        for v in self.vehicles: v.arm()

    def takeoff_all(self,alt_m:float):
        if not self._write_allowed(): raise PermissionError(self.interlock_message())
        for v in self.vehicles:
            try:
                if v.telemetry.autopilot=="ARDUPILOT": v.set_mode("GUIDED")
            except Exception: pass
            v.takeoff(alt_m)

    def enter_external_control(self):
        if not self._write_allowed(): raise PermissionError(self.interlock_message())
        self.prime_offboard(1.2)
        for v in self.vehicles:
            try:
                v.set_mode("OFFBOARD" if v.telemetry.autopilot=="PX4" else "GUIDED")
            except Exception as exc:
                self.last_error=f"{v.name}: {exc}"
        self.control_active=True

    def hold_all(self):
        for v in self.vehicles: v.desired_velocity=(0.0,0.0,0.0)
        self.control_active=True
        if self._write_allowed():
            for v in self.vehicles:
                try: v.set_mode("HOLD")
                except Exception: pass

    def land_all(self):
        if not self._write_allowed(): raise PermissionError(self.interlock_message())
        self.control_active=False
        for v in self.vehicles: v.land()

    def interlock_message(self):
        if self.mode=="REAL" and not self.real_enabled:
            return "Реальные команды заблокированы: установите SWARM_ENABLE_REAL=1 только после стендовой/SITL-проверки."
        if self.mode=="REAL" and not self.operator_confirmed:
            return "Реальные команды заблокированы: оператор не подтвердил безопасную зону."
        return "Команда заблокирована защитной логикой."

    def disconnect(self):
        self.running=False; self.control_active=False
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=1.0)
        for v in self.vehicles:
            try: v.close()
            except Exception: pass
        self.vehicles=[]; self.thread=None; self.origin_lat=None; self.origin_lon=None
