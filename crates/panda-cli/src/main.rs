//! The `panda` command-line tool. See [`panda_cli::commands::USAGE`].

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stdout = std::io::stdout();
    let stderr = std::io::stderr();
    let mut out = stdout.lock();
    let mut err = stderr.lock();
    let code = panda_cli::main_with_args(&args, &mut out, &mut err);
    let _ = out.flush();
    std::process::exit(code);
}
