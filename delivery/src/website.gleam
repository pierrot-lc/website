import app/context.{Context}
import app/web
import gleam/erlang/process
import mist
import wisp
import wisp/wisp_mist

pub fn main() {
  wisp.configure_logger()

  let secret_key_base = wisp.random_string(64)
  let ctx = Context(static_directory: static_directory())
  let handler = web.handle_request(_, ctx)

  let assert Ok(_) =
    wisp_mist.handler(handler, secret_key_base)
    |> mist.new
    |> mist.port(8000)
    |> mist.bind("0.0.0.0")
    |> mist.start_https("/etc/letsencrypt/live/pierrot-lc.dev/fullchain.pem", "/etc/letsencrypt/live/pierrot-lc.dev/privkey.pem")

  process.sleep_forever()
}

pub fn static_directory() -> String {
  let assert Ok(priv_directory) = wisp.priv_directory("website")
  priv_directory
}
