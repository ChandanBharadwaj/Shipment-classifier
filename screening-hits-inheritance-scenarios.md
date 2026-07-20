# Screening Hits — Inheritance Verification Scenarios

A set of valid, self-contained scenarios to verify screening-hit **inheritance**
logic and test cases against. Each scenario keeps a readable timeline
(`@Time t1`, `@Time t2`, …) with the *Shipment & screening info* and the
*Matches* rendered as JSON so they can be asserted directly in tests.

All scenarios use a **single keyword list `kl1`**.

> **Provenance.** Scenarios **#1–#4** are taken directly from the specification
> (#4 is the given Scenario #X, verbatim). Scenarios **#5–#10** are logically
> derived edge cases that follow from the same rules — they encode inferred
> behavior worth confirming against the implementation.

---

## 1. Domain model

We screen a **shipment** against **keywords** that belong to a **keyword list**.
Every match (a "hit") is stored as a match record in a `matches` JSON in the DB,
keyed at **shipment / keyword / keyword_list / version**. A shipment can produce
many matches across many keywords for a given list version.

A hit is either **active** (the keyword still matches the current shipment
against the current list) or **inherited** (the keyword previously matched but
has since been removed from the screening context). There are two removal causes:

| Cause | `inheritedType` | When it happens | UI |
|---|---|---|---|
| — | *(none)* | Keyword currently matches. `inherited=false`. | **Active hit** |
| Shipment edit | `SHIPMENT_UPDATE` | Keyword matched in shipment v1, then a shipment update (v2) changed the matched column so the keyword is no longer present. The keyword **still exists in the list**. | **Deleted hit — ship icon** 🚢 |
| List edit | `RISK_DATA_UPDATE` | The keyword was removed from the keyword list. The list version is bumped (e.g. `klv1` → `klv2`) and a **delta rescreen** runs for all active shipments. | **Deleted hit — info icon** ℹ️ |

### Key rules

1. **Match record fields.** A match record carries `keyword`, `keywordList`,
   `keywordListVersion`, `matchedColumn`, `inherited`, and `inheritedType`. The
   shipment identity (`shipmentNumber`, `shipmentVersion`) and the current
   screening context belong to the *Shipment & screening info*, not to the
   individual match. For regular (active) hits, `inherited=false` and
   `inheritedType=null`.
2. **Archive carry-forward.** When a keyword list is edited, the old list version
   is **archived**. The inherited match records captured against the old version
   are **carried forward** to the current/active result set. Active hits always
   come back from fresh screening; inherited hits are carried, not recomputed.
3. **RISK_DATA_UPDATE keeps the old version.** For a `RISK_DATA_UPDATE` inherited
   hit, the stored `keywordListVersion` stays at the **old** version where the
   keyword still existed (`klv1`), even though the current list version is `klv2`.
4. **`SHIPMENT_UPDATE` requires the keyword to still be in the list.** A
   `SHIPMENT_UPDATE` hit means "the keyword is still screened, but this
   shipment's column no longer contains it." Therefore a keyword that has been
   removed from the list (already `RISK_DATA_UPDATE`) can **never** later become
   `SHIPMENT_UPDATE` — it is no longer screened at all. This is the mechanical
   basis for Scenario #4 and #5.
5. **Stickiness / first-cause-wins.** Once a hit is captured as inherited, its
   `inheritedType`, `keyword`, `keywordListVersion`, and `matchedColumn` are
   fixed at capture time and are **immutable**. A later shipment update or a
   later list edit must **not** reclassify an existing inherited record. Only a
   genuine re-match (keyword back in the list **and** present in the column) can
   create a new active hit and supersede the inherited record.

---

## 2. Data shapes

Each time point carries two JSON blocks.

**Shipment & screening info** — the context the screening ran under:

| Field | Example | Notes |
|---|---|---|
| `shipmentNumber` | `s1` | The screened shipment. |
| `shipmentVersion` | `v1` | The shipment version this screening ran against. |
| `keywordList` | `kl1` | Always `kl1` in these scenarios. |
| `keywordListVersion` | `klv1` | The **current** list version screening ran against. |
| `action` | `REGULAR_SCREENING` | `REGULAR_SCREENING` \| `KEYWORD_LIST_UPDATE` \| `SHIPMENT_UPDATE` — the event that triggered this screening. |

**Matches** — an array of match records = the **full match state** for the
shipment after that step:

| Field | Example | Notes |
|---|---|---|
| `keyword` | `k1` | The matched keyword. |
| `keywordList` | `kl1` | Always `kl1` in these scenarios. |
| `keywordListVersion` | `klv1` | For `RISK_DATA_UPDATE` this is the **old** archived version where the keyword existed; may differ from the current list version in the info block. |
| `matchedColumn` | `c1` | The shipment column that produced the match. |
| `inherited` | `false` | `true` for inherited (deleted) hits. |
| `inheritedType` | `null` | `null` \| `SHIPMENT_UPDATE` \| `RISK_DATA_UPDATE`. |
| `ui` | `ACTIVE_HIT` | Derived, for readability: `ACTIVE_HIT` \| `DELETED_HIT_SHIP_ICON` \| `DELETED_HIT_INFO_ICON`. |

---

## 3. Scenarios

### Scenario #1 : Regular hit (baseline) — *from spec*

A first-time screening that produces a normal active hit.

**@Time t1 :**
With regular screening. Shipment `s1` at version `v1` is screened against `kl1`
at version `klv1`; keyword `k1` matches column `c1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

---

### Scenario #2 : Shipment update removes the matched keyword → SHIPMENT_UPDATE — *from spec*

An active hit becomes an inherited hit because the shipment was edited.

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The shipment is updated to `v2`; the edit removes `k1` from column `c1`. The
list is unchanged, so rescreening runs against `kl1@klv1`. `k1` no longer
matches, so the hit is inherited as a shipment-driven removal. `k1` still exists
in the list, which is what makes this a `SHIPMENT_UPDATE` (not `RISK_DATA_UPDATE`).
In UI we show it as a deleted hit with a ship icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v2",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "SHIPMENT_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "SHIPMENT_UPDATE",
    "ui": "DELETED_HIT_SHIP_ICON"
  }
]
```

---

### Scenario #3 : Keyword removed from list → RISK_DATA_UPDATE — *from spec*

An active hit becomes an inherited hit because the keyword list was edited.

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The keyword list is updated: `k1` is removed from `kl1`, so the current list
version becomes `klv2`. A delta rescreen is triggered for all active shipments.
The removal is captured as an inherited hit; the stored `keywordListVersion`
stays at the **old** `klv1`.
In UI we show it as a deleted hit with an info icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "KEYWORD_LIST_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "RISK_DATA_UPDATE",
    "ui": "DELETED_HIT_INFO_ICON"
  }
]
```

---

### Scenario #4 : Shipment update on a shipment that already had the keyword as inherited RISK_DATA_UPDATE (column unchanged) — *from spec (Scenario #X)*

The central stickiness case. A shipment update must **not** flip an existing
`RISK_DATA_UPDATE` inherited hit to `SHIPMENT_UPDATE`.

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The keyword list is updated: `k1` is removed and the current list version
becomes `klv2`; a delta rescreen is triggered.
In UI we show it as a deleted hit with an info icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "KEYWORD_LIST_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "RISK_DATA_UPDATE",
    "ui": "DELETED_HIT_INFO_ICON"
  }
]
```

**@Time t3 :**
The same shipment `s1` gets an update to `v2` where **nothing changed in column
`c1`**. Screening runs against the current list version `klv2`.
**Expected:** the inherited record is preserved exactly — `RISK_DATA_UPDATE`,
`k1`, `kl1`, `klv1`, `c1`. It must **not** change to `SHIPMENT_UPDATE`.
In UI we show it as a deleted hit with an info icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v2",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "SHIPMENT_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "RISK_DATA_UPDATE",
    "ui": "DELETED_HIT_INFO_ICON"
  }
]
```

---

### Scenario #5 : Shipment update on an inherited RISK_DATA_UPDATE hit where the matched column DID change — *derived*

Stickiness must hold even when the shipment edit actually touches `c1`. Because
`k1` no longer exists in `kl1` (rule 4), there is nothing to re-evaluate as a
shipment-driven removal — the original `RISK_DATA_UPDATE` record is carried
forward unchanged.

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The keyword list is updated: `k1` is removed, current list version becomes
`klv2`, delta rescreen triggered. The removal is captured as inherited.
In UI we show it as a deleted hit with an info icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "KEYWORD_LIST_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "RISK_DATA_UPDATE",
    "ui": "DELETED_HIT_INFO_ICON"
  }
]
```

**@Time t3 :**
The shipment is updated to `v2` and this time the edit **does change `c1`** (the
old value that once contained `k1` is edited away). Screening runs against
`klv2`. Since `k1` is not in the current list, no new active hit arises and the
inherited record is carried forward.
**Expected:** still `RISK_DATA_UPDATE`, `k1`, `kl1`, `klv1`, `c1`. It must
**not** become `SHIPMENT_UPDATE`.
In UI we show it as a deleted hit with an info icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v2",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "SHIPMENT_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "RISK_DATA_UPDATE",
    "ui": "DELETED_HIT_INFO_ICON"
  }
]
```

---

### Scenario #6 : Reverse ordering — SHIPMENT_UPDATE first, then the keyword is removed from the list — *derived*

The stickiness rule also holds in the other direction: an existing
`SHIPMENT_UPDATE` inherited hit must **not** be reclassified to
`RISK_DATA_UPDATE` when the keyword is later removed from the list. First cause
wins (rule 5); the delta rescreen only captures **active** hits, and this hit is
already inherited.

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The shipment is updated to `v2`; the edit removes `k1` from `c1`. The list is
unchanged (`klv1`), so the hit is inherited as a shipment-driven removal.
In UI we show it as a deleted hit with a ship icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v2",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "SHIPMENT_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "SHIPMENT_UPDATE",
    "ui": "DELETED_HIT_SHIP_ICON"
  }
]
```

**@Time t3 :**
The keyword list is now updated: `k1` is removed from `kl1`, current version
becomes `klv2`, and a delta rescreen runs. There is no active hit for `k1` on
`s1` to capture (it is already an inherited `SHIPMENT_UPDATE` record), so the
existing record is carried forward unchanged.
**Expected:** still `SHIPMENT_UPDATE`, `k1`, `kl1`, `klv1`, `c1`. It must
**not** become `RISK_DATA_UPDATE`.
In UI we show it as a deleted hit with a ship icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v2",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "KEYWORD_LIST_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "SHIPMENT_UPDATE",
    "ui": "DELETED_HIT_SHIP_ICON"
  }
]
```

---

### Scenario #7 : Multiple keywords in the same list, partial removal — *derived*

One list `kl1` has two keywords `k1` and `k2`. The shipment matches both. Only
`k1` is removed from the list. The result is a **mixed state** on a single
shipment: `k1` becomes an inherited `RISK_DATA_UPDATE` hit while `k2` remains an
active hit at the new list version.

**@Time t1 :**
With regular screening. `s1/v1` is screened against `kl1@klv1`; `k1` matches
`c1` and `k2` matches `c2`.
In UI we show two active hits.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  },
  {
    "keyword": "k2",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c2",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The keyword list is updated: only `k1` is removed from `kl1`; the current list
version becomes `klv2`; a delta rescreen runs. `k1` is captured as inherited
(`RISK_DATA_UPDATE`, old version `klv1`). `k2` still exists in the list and still
matches `c2`, so it stays active — now recorded at `klv2`.
In UI we show `k1` as a deleted hit with an info icon and `k2` as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "KEYWORD_LIST_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "RISK_DATA_UPDATE",
    "ui": "DELETED_HIT_INFO_ICON"
  },
  {
    "keyword": "k2",
    "keywordList": "kl1",
    "keywordListVersion": "klv2",
    "matchedColumn": "c2",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

---

### Scenario #8 : Keyword re-added to the list (resurrection) — *derived*

A keyword removed earlier (`RISK_DATA_UPDATE`) is later added back. If the
shipment column still contains it, screening produces a **fresh active hit** at
the new list version, and the inherited record is superseded (rule 5).

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The keyword list is updated: `k1` is removed, current version becomes `klv2`,
delta rescreen triggered. `k1` is captured as inherited (`RISK_DATA_UPDATE`,
`klv1`).
In UI we show it as a deleted hit with an info icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv2",
  "action": "KEYWORD_LIST_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "RISK_DATA_UPDATE",
    "ui": "DELETED_HIT_INFO_ICON"
  }
]
```

**@Time t3 :**
The keyword list is updated again: `k1` is **added back** to `kl1`; current
version becomes `klv3`; a delta rescreen runs. Column `c1` still contains `k1`,
so screening produces a fresh active hit at `klv3`. The prior inherited record
is superseded.
**Expected:** a single active hit for `k1` at `klv3`; no deleted/inherited
record remains for `k1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv3",
  "action": "KEYWORD_LIST_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv3",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

---

### Scenario #9 : Matched word re-introduced after a SHIPMENT_UPDATE — *derived*

A keyword that became an inherited `SHIPMENT_UPDATE` hit (because a shipment edit
removed it from the column) is later re-added to the column by a further
shipment update. Since the keyword still exists in the list, screening produces
a fresh active hit and the inherited record is cleared (rule 5).

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The shipment is updated to `v2`; the edit removes `k1` from `c1`. The list is
unchanged (`klv1`), so the hit is inherited as a shipment-driven removal.
In UI we show it as a deleted hit with a ship icon.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v2",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "SHIPMENT_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": true,
    "inheritedType": "SHIPMENT_UPDATE",
    "ui": "DELETED_HIT_SHIP_ICON"
  }
]
```

**@Time t3 :**
The shipment is updated again to `v3`; the edit **re-adds `k1` to `c1`**. The
list still contains `k1` at `klv1`, so screening produces a fresh active hit.
The prior inherited `SHIPMENT_UPDATE` record is cleared.
**Expected:** a single active hit for `k1` at `klv1`; no deleted/inherited
record remains.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v3",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "SHIPMENT_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

---

### Scenario #10 : No-op shipment update must not trigger inheritance — *derived*

A shipment update that does not touch the matched column (and no list change)
must leave the active hit active. Inheritance is only triggered by a genuine
removal — this scenario guards against false inheritance.

**@Time t1 :**
With regular screening. `s1/v1` matches `k1` in `c1` against `kl1@klv1`.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v1",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "REGULAR_SCREENING"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

**@Time t2 :**
The shipment is updated to `v2`, but the edit changes some other column and
leaves `c1` unchanged. The list is unchanged (`klv1`). `k1` still matches `c1`.
**Expected:** the hit stays active — `inherited=false`. No inherited record is
created.
In UI we show it as an active hit.

Shipment & screening info
```json
{
  "shipmentNumber": "s1",
  "shipmentVersion": "v2",
  "keywordList": "kl1",
  "keywordListVersion": "klv1",
  "action": "SHIPMENT_UPDATE"
}
```

Matches
```json
[
  {
    "keyword": "k1",
    "keywordList": "kl1",
    "keywordListVersion": "klv1",
    "matchedColumn": "c1",
    "inherited": false,
    "inheritedType": null,
    "ui": "ACTIVE_HIT"
  }
]
```

---

## 4. Coverage summary

| # | Source | Scenario | Final state of `k1` | Key assertion |
|---|---|---|---|---|
| 1 | spec | Regular hit | active | baseline active hit |
| 2 | spec | Shipment update removes keyword | `SHIPMENT_UPDATE` / `klv1` | ship icon; keyword still in list |
| 3 | spec | Keyword removed from list | `RISK_DATA_UPDATE` / `klv1` | info icon, old version kept |
| 4 | spec (#X) | Shipment update, column unchanged, over RISK_DATA_UPDATE | `RISK_DATA_UPDATE` / `klv1` | **does not flip** to SHIPMENT_UPDATE |
| 5 | derived | Shipment update, column changed, over RISK_DATA_UPDATE | `RISK_DATA_UPDATE` / `klv1` | stickiness holds even when column moves |
| 6 | derived | SHIPMENT_UPDATE first, then list removal | `SHIPMENT_UPDATE` / `klv1` | **does not flip** to RISK_DATA_UPDATE |
| 7 | derived | Partial removal, two keywords | `k1` inherited, `k2` active | mixed state on one shipment |
| 8 | derived | Keyword re-added to list | active / `klv3` | inherited record superseded |
| 9 | derived | Word re-introduced after SHIPMENT_UPDATE | active / `klv1` | inherited record cleared |
| 10 | derived | No-op shipment update | active / `klv1` | inheritance not triggered |
