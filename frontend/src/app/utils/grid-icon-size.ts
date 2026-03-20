const ICON_SIZE_GOAL_WIDTH: Record<string, number> = {
  XS: 25,
  S: 50,
  M: 80,
  L: 130,
  XL: 200,
};

export function iconSizeToGoalWidth(size: string): number {
  return ICON_SIZE_GOAL_WIDTH[size] ?? ICON_SIZE_GOAL_WIDTH['M'];
}
