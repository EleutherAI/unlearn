# Orth Circuit Breaker HP Tuning

Goal: Find boundary between minimal capability impact and noticeable damage.

Defaults: `remove_coef=23`, `orth_coef=10`, `retain_coef=2`

## Planned Runs

| Run | remove_coef | orth_coef | Rationale |
|-----|-------------|-----------|-----------|
| 1   | 15          | 5         | Low intervention baseline |
| 2   | 23          | 10        | Default values |
| 3   | 30          | 15        | Moderately higher |
| 4   | 40          | 20        | High intervention (expect damage) |

## Results

| remove_coef | orth_coef | WMDP (lower=better) | MMLU (higher=better) | Job ID  | Notes |
|-------------|-----------|---------------------|----------------------|---------|-------|
| 15          | 5         | pending             | pending              | 2018391 |       |
| 23          | 10        | pending             | pending              | 2018392 |       |
| 30          | 15        | pending             | pending              | 2018393 |       |
| 40          | 20        | pending             | pending              | 2018394 |       |
