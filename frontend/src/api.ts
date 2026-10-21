export const CONNECTION_EVENT = "agmen:connection-error";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly path: string,
    public readonly status?: number,
    public readonly transport = false,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type RequestOptions = { timeoutMs?: number; retries?: number };
const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function authHeaders(): Record<string, string> {
  const token = sessionStorage.getItem("agmen-token");
  return token ? { "X-AGMEN-Token": token } : {};
}

function reportConnectionError(error: ApiError) {
  if (error.transport || error.status === 401) {
    window.dispatchEvent(new CustomEvent(CONNECTION_EVENT, { detail: error }));
  }
}

async function httpError(response: Response, path: string): Promise<ApiError> {
  let detail = "";
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") detail = payload.detail;
    else if (Array.isArray(payload.detail)) {
      detail = payload.detail.map((item: { loc?: string[]; msg?: string }) =>
        `${item.loc?.filter((part) => part !== "body").join(".") || "параметры"}: ${item.msg || "недопустимое значение"}`,
      ).join("; ");
    }
  } catch { /* A reverse proxy may return an HTML error page. */ }
  const reason = response.status === 401
    ? "Сервер доступен. Укажите действующий токен в настройках платформы."
    : response.status === 403
      ? "Сервер запретил действие. Проверьте права и адрес приложения."
      : response.status === 422
        ? "Проверьте параметры запроса."
        : response.status === 429
          ? "Очередь расчётов заполнена. Дождитесь завершения текущей задачи."
          : response.status >= 500
            ? "Сервер не смог выполнить запрос."
            : "Запрос не выполнен.";
  return new ApiError(`${reason}${detail && response.status !== 401 ? ` ${detail}` : ""} (HTTP ${response.status}, ${path})`, path, response.status);
}

async function request<T>(
  path: string,
  body: unknown,
  read: (response: Response) => Promise<T>,
  options: RequestOptions = {},
): Promise<T> {
  // Only relative API routes are accepted, so credentials stay on this origin.
  if (!path.startsWith("/api/") || path.startsWith("//")) {
    throw new ApiError("Некорректный адрес API", path);
  }
  const isRead = body === undefined;
  const attempts = 1 + (isRead ? (options.retries ?? 1) : 0);
  for (let attempt = 0; attempt < attempts; attempt++) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs ?? (isRead ? 8000 : 20000));
    let failure: ApiError | undefined;
    try {
      const response = await fetch(path, {
        method: isRead ? "GET" : "POST",
        headers: { ...authHeaders(), ...(isRead ? {} : { "Content-Type": "application/json" }) },
        body: isRead ? undefined : JSON.stringify(body),
        signal: controller.signal,
        cache: "no-store",
        credentials: "same-origin",
      });
      if (!response.ok) throw await httpError(response, path);
      return await read(response);
    } catch (error) {
      if (error instanceof ApiError) failure = error;
      else {
        const reason = controller.signal.aborted ? "Сервер не ответил вовремя" : "Нет соединения с локальным сервером";
        const instruction = isRead
          ? "Запустите RUN_AGMEN.cmd. Подключение восстановится автоматически."
          : "Результат команды неизвестен. После восстановления связи проверьте список задач перед повторным запуском.";
        failure = new ApiError(`${reason} (${path}). ${instruction}`, path, undefined, true);
      }
      reportConnectionError(failure);
      const retryable = failure.transport || [502, 503, 504].includes(failure.status ?? 0);
      if (!retryable || attempt === attempts - 1) throw failure;
    } finally {
      window.clearTimeout(timeout);
    }
    await pause(350 * (attempt + 1));
  }
  throw new ApiError("Запрос не выполнен", path);
}

export function api<T>(path: string, body?: unknown, options?: RequestOptions): Promise<T> {
  return request(path, body, (response) => response.json() as Promise<T>, options);
}

export const requestJSON = api;

export function requestBlob(path: string): Promise<Blob> {
  return request(path, undefined, (response) => response.blob(), { timeoutMs: 30000 });
}

export async function download(path: string, name: string) {
  const blob = await requestBlob(path);
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 3000);
}
