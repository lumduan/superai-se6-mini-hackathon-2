# 🏆 [Super AI Engineer S6] OCR ผลเลือกตั้ง (รอบสอง)

## 🧾 Overview

Extract structured voting data from scanned Thai election result documents (Form สส.6/1) from the 2026 Thai general election.

Given PNG scans of official election documents, your task is to:

1. Locate the correct row for each party in the voting tables
2. Extract the corresponding vote count
3. Convert all numbers to Arabic digits (0–9)

Party names are pre-filled in the submission template — you only need to predict the vote counts.

> ⚠️ **This is a test-set-only competition**
> There is no training data. You are expected to use:
> - OCR tools
> - Vision LLMs
> - APIs
>
> Focus areas:
> - Prompt engineering
> - Post-processing
> - Pipeline design

---

## 📌 Description

### 📝 IMPORTANT (21/03/2026)

ส่งคำตอบเฉพาะ:

```
id,votes
```

- ไม่ต้องส่ง field อื่น
- ไม่ต้องสนใจ `party_name`
- เรียงตามลำดับจากบนลงล่าง

---

## 🎯 Task

ดึงจำนวนคะแนนเสียง (vote count) จากเอกสารผลการเลือกตั้งที่เป็นภาพสแกน

สำหรับแต่ละแถว:
- ทำนายค่า `votes` ของ `id` ที่กำหนด

Public template มี metadata เพื่อช่วย align OCR  
แต่ key ที่ใช้จริงในการ submit คือ `id`

---

## 📂 Data

- 📄 300 documents
- 🖼️ 846 PNG images
- 🧮 10,053 submission rows

### 📁 Public Reference Files

- `submission_template_v3.csv`

### 📝 Notes

- มีบางแถวใน template เดิมใช้ placeholder เช่น:
  - `empty`
  - `UNKNOWN`
  - `Unknown Party`
- Template ใหม่มีการ normalize แล้ว
- ✅ submission `id` ไม่เปลี่ยน

---

## 📤 Submission Format

ส่งไฟล์ CSV ที่มีแค่:

```
id,votes
```

### 📏 Rules

| Rule | Detail |
|------|--------|
| ❌ | ห้ามแก้ไข `id` |
| ✅ | ต้องส่งให้ครบทุกแถว |
| 🔢 | `votes` ต้องเป็นเลขอารบิกเท่านั้น (0–9) |
| 🚫 | `party_name` ไม่ใช่ส่วนของ submission |

---

## 📊 Evaluation

**Metric: Levenshtein Distance**

เปรียบเทียบ `votes` ที่ทำนายกับค่าจริง โดยใช้:

> จำนวนการแก้ไขขั้นต่ำ (insert / delete / substitute) ที่ทำให้ string เท่ากัน

### 🧮 Formula

$$\text{Final Score} = \text{mean Levenshtein distance across all rows}$$

- 🎯 ค่ายิ่งต่ำยิ่งดี
- ✅ Perfect score = **0**

### 📌 Scoring Examples

| Actual Votes | Predicted Votes | Distance |
|:---:|:---:|:---:|
| `"14813"` | `"14813"` | 0 |
| `"14813"` | `"14812"` | 1 |
| `"14813"` | `"1481"` | 1 |
| `"14813"` | `"0"` | 5 |

---

## ⚠️ Important Notes

- `party_name` ใช้เพื่อ reference เท่านั้น
- ❌ grader ไม่ใช้ `party_name` ในการให้คะแนน

---

## 🔄 Update (2026-03-21)

มีการอัปเดต public template เนื่องจาก:
- มี placeholder เช่น:
  - `empty`
  - `UNKNOWN`
  - `Unknown Party`
  - numbered placeholders

### ✅ การเปลี่ยนแปลง

- ปรับ normalization ของข้อมูลอ้างอิง

### ❌ ไม่มีผลต่อ

- Submission format
- Submission IDs
- Evaluation metric

---

## � Competition Rules

- ไม่มี training data หรือ ground truth — ให้ใช้ tools และ models ที่มีอยู่
- อนุญาตให้ใช้ OCR tool, Vision LLM, หรือ API ใดก็ได้
- รับทั้ง automated และ semi-automated approaches
- แต่ละคนสามารถ submit ได้สูงสุด **4 ครั้งต่อวัน**

---

## �🚀 Summary

คุณต้อง:

1. อ่าน OCR จากภาพ
2. หา row ที่ถูกต้อง
3. Extract ตัวเลข
4. Normalize เป็นเลข 0–9
5. Submit `id,votes` ให้ครบ
