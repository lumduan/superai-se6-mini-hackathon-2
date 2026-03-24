# Phase 2: False Positive table detection from dark fold lines — fix with intersection count ≥ 6

## 🐛 Problem

`has_table_structure()` detects tables by counting horizontal and vertical line pixels. A non-table page (e.g. cover page) with dark fold creases in both directions easily passes the threshold:

```python
return h_count > 500 and v_count > 200  # fold lines pass this trivially
```

### Downstream Impact
Cover page → flagged as table → OCR on wrong page → district codes/dates parsed as votes → wrong `submission.csv` → Levenshtein score increases

## ✅ Fix — Intersection Count ≥ 6

Real tables have many H×V intersections. Fold lines produce at most 1–2.

```python
h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
intersections = cv2.bitwise_and(h_lines, v_lines)
return cv2.countNonZero(intersections) >= 6
```

Form สส.6/1 minimum: 4 columns × 3 rows = **8 intersections**. Threshold 6 is safely conservative.

| Scenario | Before | After |
|---|---|---|
| Real vote table | ✅ | ✅ |
| Fold lines only | ❌ False Positive | ✅ Rejected |
| Border lines only | ❌ False Positive | ✅ Rejected |

## 📁 File
`src/phase2_detection/detection.py` — `has_table_structure()`

## 🔗 Related
- [master-plan.md Phase 2](https://github.com/lumduan/superai-se6-mini-hackathon-2/blob/main/docs/plan/master-plan.md)