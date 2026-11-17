# Blurring and unblurring sensitive images

## Background

Planned in #5689, and shipped: images an ingest
service marks as sensitive are drawn blurred until someone overrules them,
with a hover peek and a Blur / Unblur button. The design now lives in the
permanent docs: "Blur state" in `docs/ARCHITECTURE.md` (the state model,
where the overrides live, why it is no access control), "Blurred images" in
`docs/FRONTEND.md` (the state service, `vtBlur`, the focused viewers, the
Browse canvas), and "Blurring sensitive images" in `docs/DEPLOYMENT.md`
(the rules, for operators).

## Open questions

- **Faces and their source photo.** Both inherit the same ingest fields, so
  both start blurred, but an override on one does not move the other. Linking
  them is possible through the source-item link if that proves annoying.
- **The document viewer.** A document's poster thumbnail is blurred with the
  other stills; the viewer itself (an embedded `<object>`) is left sharp.
- **Blur state as an export column.** Results exporters could offer the
  effective state as a column; nothing has asked for it yet.
