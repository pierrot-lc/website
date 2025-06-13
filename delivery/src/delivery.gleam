import app/web
import argv
import gleam/erlang/process
import gleam/io
import mist
import wisp
import wisp/wisp_mist

pub fn start_https(static_dir: String, certfile: String, keyfile: String) {
  wisp.configure_logger()
  let secret_key_base = wisp.random_string(64)
  let handler = web.handle_request(_, static_dir)
  let assert Ok(_) =
    wisp_mist.handler(handler, secret_key_base)
    |> mist.new
    |> mist.port(8000)
    |> mist.bind("0.0.0.0")
    |> mist.start_https(certfile, keyfile)

  process.sleep_forever()
}

pub fn start_http(static_dir: String) {
  wisp.configure_logger()
  let secret_key_base = wisp.random_string(64)
  let handler = web.handle_request(_, static_dir)
  let assert Ok(_) =
    wisp_mist.handler(handler, secret_key_base)
    |> mist.new
    |> mist.port(8000)
    |> mist.bind("0.0.0.0")
    |> mist.start_http

  process.sleep_forever()
}

pub fn main() {
  case argv.load().arguments {
    [static_dir, certfile, keyfile] ->
      start_https(static_dir, certfile, keyfile)
    [static_dir] -> start_http(static_dir)
    _ -> io.println("usage: ./delivery <static_dir> <certfile> <keyfile>")
  }
}
