# Dataset Preprocessing & Validation Report

- **Generated At:** 2026-10-27T14:45:17.092668+00:00
- **Source File:** `C:\Users\ksais\OneDrive\Desktop\MLOPS_PROJECT\ChatBot-PipeLine\data\raw\Intent.json`
- **Random Seed:** `42`
- **Total Raw Intents:** 22
- **Total Raw Utterances:** 143
- **Total Cleaned Unique Utterances:** 142
- **Intra-Intent Duplicates Deduplicated:** 1
- **Cross-Intent Conflicts Detected:** 0

## Split Sizes
- **Train Set:** 98 examples (69.0%)
- **Validation Set:** 22 examples (15.5%)
- **Test Set:** 22 examples (15.5%)

## Intent Class Distribution across Splits
| Class ID | Intent Name | Total | Train | Val | Test |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 0 | `Clever` | 7 | 5 | 1 | 1 |
| 1 | `CourtesyGoodBye` | 6 | 4 | 1 | 1 |
| 2 | `CourtesyGreeting` | 7 | 5 | 1 | 1 |
| 3 | `CourtesyGreetingResponse` | 8 | 6 | 1 | 1 |
| 4 | `CurrentHumanQuery` | 7 | 5 | 1 | 1 |
| 5 | `GoodBye` | 4 | 2 | 1 | 1 |
| 6 | `Gossip` | 6 | 4 | 1 | 1 |
| 7 | `Greeting` | 7 | 5 | 1 | 1 |
| 8 | `GreetingResponse` | 8 | 6 | 1 | 1 |
| 9 | `Jokes` | 6 | 4 | 1 | 1 |
| 10 | `NameQuery` | 6 | 4 | 1 | 1 |
| 11 | `NotTalking2U` | 7 | 5 | 1 | 1 |
| 12 | `PodBayDoor` | 7 | 5 | 1 | 1 |
| 13 | `PodBayDoorResponse` | 7 | 5 | 1 | 1 |
| 14 | `RealNameQuery` | 6 | 4 | 1 | 1 |
| 15 | `SelfAware` | 7 | 5 | 1 | 1 |
| 16 | `Shutup` | 7 | 5 | 1 | 1 |
| 17 | `Swearing` | 4 | 2 | 1 | 1 |
| 18 | `Thanks` | 6 | 4 | 1 | 1 |
| 19 | `TimeQuery` | 7 | 5 | 1 | 1 |
| 20 | `UnderstandQuery` | 6 | 4 | 1 | 1 |
| 21 | `WhoAmI` | 6 | 4 | 1 | 1 |

## Validation & Split Warnings
- ⚠️ Intent 'UnderstandQuery' is missing optional field 'entityType'. Will default to 'NA'.
