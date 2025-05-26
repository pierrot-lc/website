import app/web
import argv
import gleam/erlang/process
import gleam/io
import mist
import wisp
import wisp/wisp_mist

pub fn start(certfile: String, keyfile: String) {
  wisp.configure_logger()
  let secret_key_base = wisp.random_string(64)
  let assert Ok(_) =
    wisp_mist.handler(web.handle_request, secret_key_base)
    |> mist.new
    |> mist.port(8000)
    |> mist.bind("0.0.0.0")
    |> mist.start_https(certfile, keyfile)

  process.sleep_forever()
}

pub fn main() {
  case argv.load().arguments {
    [certfile, keyfile] -> start(certfile, keyfile)
    _ -> io.println("usage: gleam run -- <certfile> <keyfile>")
  }
}
