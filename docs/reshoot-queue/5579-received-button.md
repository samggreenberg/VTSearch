# #5579: Received… in the Detectors card header, and the Share / Received dialogs

The **Received…** button shows only on a deployment with `detector_sharing_source`
set, which the screenshot fixtures leave off, so these shots should come out
unchanged; they are listed because the card header is where it would show if a
fixture turns sharing on. The two dialogs are new surfaces with no shot yet: the
user-guide issue that follows (#5580) adds theirs.

- `dashboard-loaded` — the Detectors card header gains **Received…** beside the **+**
- `dashboard-manage` — the same header, with a row selected
