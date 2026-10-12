/** A count with its noun: "1 positive", "3 positives". */
export function countPhrase(n: number, one: string, many: string): string {
  return `${n} ${n === 1 ? one : many}`;
}
