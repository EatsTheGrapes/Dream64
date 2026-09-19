#[cfg(not(windows))]
use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

#[derive(Debug)]
pub(crate) struct LaunchOptions {
    pub(crate) skin: Option<PathBuf>,
    pub(crate) world: Option<PathBuf>,
    pub(crate) map: Option<PathBuf>,
    pub(crate) connect: SocketAddr,
    pub(crate) record: Option<PathBuf>,
    pub(crate) replay: Option<PathBuf>,
    pub(crate) startup_replay: Option<PathBuf>,
}

impl LaunchOptions {
    pub(crate) fn parse() -> Result<Self, String> {
        let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
        if arguments.is_empty() {
            let address = prompt_for_server()?;
            Self::parse_from(["--connect".into(), address.to_string().into()])
        } else {
            Self::parse_from(arguments)
        }
    }

    pub(crate) fn parse_from(arguments: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Self, String> {
        let mut options = Self {
            skin: None,
            world: None,
            map: None,
            connect: "127.0.0.1:51664"
                .parse()
                .expect("default loopback address is valid"),
            record: None,
            replay: None,
            startup_replay: None,
        };
        let mut connect_seen = false;
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.to_string_lossy().as_ref() {
                "--skin" => set_path_option(
                    &mut options.skin,
                    next_path(&mut arguments, "--skin")?,
                    "--skin",
                )?,
                "--world" => set_path_option(
                    &mut options.world,
                    next_path(&mut arguments, "--world")?,
                    "--world",
                )?,
                "--map" => set_path_option(
                    &mut options.map,
                    next_path(&mut arguments, "--map")?,
                    "--map",
                )?,
                "--connect" => {
                    if std::mem::replace(&mut connect_seen, true) {
                        return Err("--connect may only be specified once".to_owned());
                    }
                    let value = arguments
                        .next()
                        .ok_or_else(|| "--connect requires an IP address and port".to_owned())?;
                    options.connect = value
                        .to_string_lossy()
                        .parse::<SocketAddr>()
                        .map_err(|error| format!("invalid --connect address: {error}"))?;
                }
                "--record-replay" => set_path_option(
                    &mut options.record,
                    next_path(&mut arguments, "--record-replay")?,
                    "--record-replay",
                )?,
                "--replay" => set_path_option(
                    &mut options.replay,
                    next_path(&mut arguments, "--replay")?,
                    "--replay",
                )?,
                "--startup-replay" => set_path_option(
                    &mut options.startup_replay,
                    next_path(&mut arguments, "--startup-replay")?,
                    "--startup-replay",
                )?,
                other if other.ends_with(".dmf") && options.skin.is_none() => {
                    options.skin = Some(PathBuf::from(argument));
                }
                other => {
                    return Err(format!(
                        "unknown client argument {other:?}; use [--connect 127.0.0.1:51664] [--startup-replay <file>] [--record-replay <file>] or --replay <file>, or --world <.dme> --map <.dmm> [--skin <.dmf>]"
                    ));
                }
            }
        }
        if options.map.is_some() && options.world.is_none() {
            return Err("--map requires --world".to_owned());
        }
        if options.record.is_some() && options.replay.is_some() {
            return Err("--record-replay and --replay are mutually exclusive".to_owned());
        }
        if options.replay.is_some() && options.startup_replay.is_some() {
            return Err("--replay and --startup-replay are mutually exclusive".to_owned());
        }
        if options.world.is_some() && (options.record.is_some() || options.replay.is_some()) {
            return Err("replay recording/playback cannot be combined with --world".to_owned());
        }
        if options.world.is_some() && options.startup_replay.is_some() {
            return Err("--startup-replay cannot be combined with --world".to_owned());
        }
        Ok(options)
    }
}

#[cfg(windows)]
pub(crate) fn prompt_for_server() -> Result<SocketAddr, String> {
    const SCRIPT: &str = r#"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$form = New-Object System.Windows.Forms.Form
$form.Text = 'Connect to Dream64'
$form.ClientSize = New-Object System.Drawing.Size(360, 165)
$form.FormBorderStyle = 'FixedDialog'
$form.StartPosition = 'CenterScreen'
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.TopMost = $true

$ipLabel = New-Object System.Windows.Forms.Label
$ipLabel.Text = 'Server IP address'
$ipLabel.Location = New-Object System.Drawing.Point(20, 18)
$ipLabel.AutoSize = $true
$form.Controls.Add($ipLabel)

$ip = New-Object System.Windows.Forms.TextBox
$ip.Location = New-Object System.Drawing.Point(20, 40)
$ip.Size = New-Object System.Drawing.Size(320, 23)
$form.Controls.Add($ip)

$portLabel = New-Object System.Windows.Forms.Label
$portLabel.Text = 'Port'
$portLabel.Location = New-Object System.Drawing.Point(20, 75)
$portLabel.AutoSize = $true
$form.Controls.Add($portLabel)

$port = New-Object System.Windows.Forms.NumericUpDown
$port.Location = New-Object System.Drawing.Point(20, 97)
$port.Size = New-Object System.Drawing.Size(120, 23)
$port.Minimum = 1
$port.Maximum = 65535
$port.Value = 51664
$form.Controls.Add($port)

$connect = New-Object System.Windows.Forms.Button
$connect.Text = 'Connect'
$connect.Location = New-Object System.Drawing.Point(178, 96)
$connect.Size = New-Object System.Drawing.Size(78, 26)
$connect.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.AcceptButton = $connect
$form.Controls.Add($connect)

$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = 'Cancel'
$cancel.Location = New-Object System.Drawing.Point(262, 96)
$cancel.Size = New-Object System.Drawing.Size(78, 26)
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.CancelButton = $cancel
$form.Controls.Add($cancel)

$form.Add_Shown({ $ip.Focus() })
if ($form.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($ip.Text.Trim() + '|' + [int]$port.Value)
    exit 0
}
exit 1
"#;
    loop {
        let output = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-STA", "-Command", SCRIPT])
            .output()
            .map_err(|error| format!("could not open connection window: {error}"))?;
        if !output.status.success() {
            return Err("connection cancelled".to_owned());
        }
        let response = String::from_utf8(output.stdout)
            .map_err(|_| "connection window returned invalid text".to_owned())?;
        let (ip, port) = response
            .trim()
            .split_once('|')
            .ok_or("connection window returned an invalid address")?;
        match (ip.parse::<IpAddr>(), port.parse::<u16>()) {
            (Ok(ip), Ok(port @ 1..)) => return Ok(SocketAddr::new(ip, port)),
            _ => {
                let _ = std::process::Command::new("powershell.exe")
                    .args([
                        "-NoProfile",
                        "-STA",
                        "-Command",
                        "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Enter a valid server IP address.', 'Dream64', 'OK', 'Warning')",
                    ])
                    .status();
            }
        }
    }
}

#[cfg(not(windows))]
pub(crate) fn prompt_for_server() -> Result<SocketAddr, String> {
    println!("Dream64 Server Connection\n");
    let ip = loop {
        let value = prompt_line("Server IP address: ")?;
        match value.parse::<IpAddr>() {
            Ok(ip) => break ip,
            Err(error) => eprintln!("Invalid IP address: {error}"),
        }
    };
    let port = loop {
        let value = prompt_line("Server port [51664]: ")?;
        if value.is_empty() {
            break 51_664;
        }
        match value.parse::<u16>() {
            Ok(0) => eprintln!("Port must be between 1 and 65535."),
            Ok(port) => break port,
            Err(error) => eprintln!("Invalid port: {error}"),
        }
    };
    Ok(SocketAddr::new(ip, port))
}

#[cfg(not(windows))]
fn prompt_line(label: &str) -> Result<String, String> {
    print!("{label}");
    std::io::stdout()
        .flush()
        .map_err(|error| format!("could not display connection prompt: {error}"))?;
    let mut value = String::new();
    std::io::stdin()
        .read_line(&mut value)
        .map_err(|error| format!("could not read connection prompt: {error}"))?;
    Ok(value.trim().to_owned())
}

pub(crate) fn set_path_option(slot: &mut Option<PathBuf>, value: PathBuf, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("{flag} may only be specified once"));
    }
    Ok(())
}

pub(crate) fn next_path(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<PathBuf, String> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} requires a path"))
}