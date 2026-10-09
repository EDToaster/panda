//! Where the *File* menu gets paths from: a [`PathPicker`].
//!
//! * [`NativePicker`] (cargo feature `dialogs`, on by default): the desktop's file dialog via
//!   `rfd` (XDG desktop portal on Linux). The dialog runs on a worker thread and the answer is
//!   polled each frame, so the UI keeps drawing (and the sim thread is never involved).
//! * [`PromptPicker`]: an in-app text field (no desktop integration; used without `dialogs`).
//! * [`ScriptedPicker`]: answers from a queue (tests inject paths with it; no GPU, no portal).
//!
//! A picker is asynchronous: [`PathPicker::start`] begins a request and [`PathPicker::poll`]
//! eventually returns it with the chosen path (`None` = cancelled).

use std::cell::RefCell;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::rc::Rc;

/// What a path is wanted for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PickPurpose {
    /// Open a `.panda.json` document.
    Open,
    /// Save the document under a new path.
    SaveAs,
    /// Read a hex image for a ROM.
    ImportRom,
}

impl PickPurpose {
    /// Dialog title.
    pub fn title(self) -> &'static str {
        match self {
            PickPurpose::Open => "Open design",
            PickPurpose::SaveAs => "Save design as",
            PickPurpose::ImportRom => "Import ROM from hex file",
        }
    }

    /// File-type filter `(name, extensions)` of the dialog.
    pub fn filter(self) -> (&'static str, &'static [&'static str]) {
        match self {
            PickPurpose::Open | PickPurpose::SaveAs => ("Panda design", &["json"]),
            PickPurpose::ImportRom => ("Hex image", &["hex", "mem", "txt"]),
        }
    }
}

/// One request for a path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PickRequest {
    /// Purpose.
    pub purpose: PickPurpose,
    /// Suggested path (current file for Save as, directory of the last file, …).
    pub suggested: Option<PathBuf>,
}

/// A source of paths for the *File* menu (see the module docs).
pub trait PathPicker {
    /// Begin `req`. A request already in progress is replaced.
    fn start(&mut self, req: PickRequest);
    /// The finished request and its answer (`None` path = cancelled), once.
    fn poll(&mut self) -> Option<(PickRequest, Option<PathBuf>)>;
    /// True while a request is in progress.
    fn busy(&self) -> bool;
    /// The in-app prompt to draw, for pickers that need one.
    fn prompt(&mut self) -> Option<&mut PathPrompt> {
        None
    }
}

/// The picker for the app: [`NativePicker`] with the `dialogs` feature, else [`PromptPicker`].
pub fn default_picker() -> Box<dyn PathPicker> {
    #[cfg(feature = "dialogs")]
    {
        Box::new(NativePicker::default())
    }
    #[cfg(not(feature = "dialogs"))]
    {
        Box::new(PromptPicker::default())
    }
}

// ---------------------------------------------------------------- in-app prompt

/// State of an in-app path prompt (drawn by [`super::menu`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathPrompt {
    /// The request.
    pub req: PickRequest,
    /// Text in the field.
    pub text: String,
    /// Set by the UI: `Some(true)` = OK, `Some(false)` = cancel.
    pub answer: Option<bool>,
}

/// Asks for paths with a text field inside the app.
#[derive(Debug, Default)]
pub struct PromptPicker {
    prompt: Option<PathPrompt>,
}

impl PathPicker for PromptPicker {
    fn start(&mut self, req: PickRequest) {
        let text = req
            .suggested
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        self.prompt = Some(PathPrompt {
            req,
            text,
            answer: None,
        });
    }

    fn poll(&mut self) -> Option<(PickRequest, Option<PathBuf>)> {
        let ok = self.prompt.as_ref()?.answer?;
        let p = self.prompt.take()?;
        let text = p.text.trim();
        let path = (ok && !text.is_empty()).then(|| PathBuf::from(text));
        Some((p.req, path))
    }

    fn busy(&self) -> bool {
        self.prompt.is_some()
    }

    fn prompt(&mut self) -> Option<&mut PathPrompt> {
        self.prompt.as_mut()
    }
}

// ---------------------------------------------------------------- scripted (tests)

/// Answers requests from a shared queue (clone it to keep a handle for pushing answers).
/// With an empty queue a request stays pending.
#[derive(Clone, Debug, Default)]
pub struct ScriptedPicker {
    /// Answers in order (`None` = the user cancels).
    pub answers: Rc<RefCell<VecDeque<Option<PathBuf>>>>,
    /// Every request seen, in order.
    pub requests: Rc<RefCell<Vec<PickRequest>>>,
    current: Option<PickRequest>,
}

impl ScriptedPicker {
    /// Queue the answer for the next request.
    pub fn push(&self, answer: Option<PathBuf>) {
        self.answers.borrow_mut().push_back(answer);
    }
}

impl PathPicker for ScriptedPicker {
    fn start(&mut self, req: PickRequest) {
        self.requests.borrow_mut().push(req.clone());
        self.current = Some(req);
    }

    fn poll(&mut self) -> Option<(PickRequest, Option<PathBuf>)> {
        self.current.as_ref()?;
        let answer = self.answers.borrow_mut().pop_front()?;
        Some((self.current.take()?, answer))
    }

    fn busy(&self) -> bool {
        self.current.is_some()
    }
}

// ---------------------------------------------------------------- native (rfd)

/// The desktop's file dialog (`rfd`), shown from a worker thread.
#[cfg(feature = "dialogs")]
#[derive(Debug, Default)]
pub struct NativePicker {
    job: Option<(PickRequest, std::sync::mpsc::Receiver<Option<PathBuf>>)>,
}

#[cfg(feature = "dialogs")]
impl PathPicker for NativePicker {
    fn start(&mut self, req: PickRequest) {
        let (tx, rx) = std::sync::mpsc::channel();
        let r = req.clone();
        let spawned = std::thread::Builder::new()
            .name("panda-file-dialog".into())
            .spawn(move || {
                let (name, ext) = r.purpose.filter();
                let mut d = rfd::FileDialog::new()
                    .set_title(r.purpose.title())
                    .add_filter(name, ext);
                if let Some(s) = &r.suggested {
                    if s.is_dir() {
                        d = d.set_directory(s);
                    } else {
                        if let Some(dir) = s.parent().filter(|d| !d.as_os_str().is_empty()) {
                            d = d.set_directory(dir);
                        }
                        if let Some(f) = s.file_name() {
                            d = d.set_file_name(f.to_string_lossy());
                        }
                    }
                }
                let path = match r.purpose {
                    PickPurpose::SaveAs => d.save_file(),
                    PickPurpose::Open | PickPurpose::ImportRom => d.pick_file(),
                };
                let _ = tx.send(path);
            });
        if spawned.is_ok() {
            self.job = Some((req, rx));
        }
    }

    fn poll(&mut self) -> Option<(PickRequest, Option<PathBuf>)> {
        let (_, rx) = self.job.as_ref()?;
        let answer = match rx.try_recv() {
            Ok(p) => p,
            Err(std::sync::mpsc::TryRecvError::Empty) => return None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        };
        let (req, _) = self.job.take()?;
        Some((req, answer))
    }

    fn busy(&self) -> bool {
        self.job.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(p: PickPurpose) -> PickRequest {
        PickRequest {
            purpose: p,
            suggested: Some("/a/b.panda.json".into()),
        }
    }

    #[test]
    fn prompt_picker_answers() {
        let mut p = PromptPicker::default();
        p.start(req(PickPurpose::SaveAs));
        assert!(p.busy() && p.poll().is_none());
        assert_eq!(p.prompt().unwrap().text, "/a/b.panda.json");
        p.prompt().unwrap().answer = Some(true);
        let (r, path) = p.poll().unwrap();
        assert_eq!(r.purpose, PickPurpose::SaveAs);
        assert_eq!(path.unwrap(), PathBuf::from("/a/b.panda.json"));
        assert!(!p.busy());
        p.start(req(PickPurpose::Open));
        p.prompt().unwrap().answer = Some(false);
        assert_eq!(p.poll().unwrap().1, None);
    }

    #[test]
    fn scripted_picker_waits_for_answers() {
        let s = ScriptedPicker::default();
        let mut p = s.clone();
        p.start(req(PickPurpose::Open));
        assert!(p.poll().is_none() && p.busy());
        s.push(Some("x".into()));
        assert_eq!(p.poll().unwrap().1, Some(PathBuf::from("x")));
        assert_eq!(s.requests.borrow().len(), 1);
        assert!(p.poll().is_none());
    }
}
