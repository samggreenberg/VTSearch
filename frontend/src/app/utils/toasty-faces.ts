/** King Toasty's faces (#4680): the pictures in `frontend/public/`. */
export type ToastyFace = 'happy' | 'sad' | 'surprised';

export const TOASTY_FACES: Readonly<Record<ToastyFace, string>> = {
  happy: 'logo.png',
  sad: 'toasty-sad.png',
  surprised: 'toasty-surprised.png',
};
