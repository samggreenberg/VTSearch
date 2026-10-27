# #5320: Face is offered only on a server with face models

The app downloads no face models any more, and the screenshot harness supplies
none, so Face drops out of the import pickers: an image source no longer offers
a Face output beside Image. `image2face`'s form also loses its Face provider,
Find sideways faces and `similarity` options, though no shot opens it.

- `import-advanced` — the kinds-of-media picker for a photo source no longer lists Face
