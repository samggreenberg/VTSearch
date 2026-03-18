const ICON_SIZE_GOAL_WIDTH: Record<string, number> = {
  XS: 40,
  S: 60,
  M: 80,
  L: 120,
  XL: 180,
};

export function iconSizeToGoalWidth(size: string): number {
  return ICON_SIZE_GOAL_WIDTH[size] ?? ICON_SIZE_GOAL_WIDTH['M'];
}
