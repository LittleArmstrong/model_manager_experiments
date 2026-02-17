---
# 🚀 Schnellstart

## 1️⃣ run.bat ausführen

1. Öffne den Projektordner.
2. Doppelklicke auf `run.bat`.
3. Das Training startet automatisch.

Fertig ✅
---

# ❗ Falls run.bat nicht funktioniert

Wenn ein Fehler erscheint oder sich nichts startet, bitte folgende Schritte ausführen:

---

# 🛠 Alternative Methode mit VS Code

## 1️⃣ Projekt in VS Code öffnen

- Rechtsklick auf den Projektordner
- **"Mit Code öffnen"** auswählen

ODER

- VS Code starten
- **Datei → Ordner öffnen**
- Projektordner auswählen

---

## 2️⃣ Terminal öffnen

In VS Code:

- Menü: **Terminal → Neues Terminal**
- Es öffnet sich unten ein Terminal-Fenster

Du kannst dort zwischen **CMD**, **PowerShell** oder **Git Bash** wählen.

---

## 3️⃣ Virtuelle Umgebung aktivieren

### 🖥 Wenn du CMD benutzt:

```cmd
.venv\Scripts\activate.bat
```

---

### 💙 Wenn du PowerShell benutzt:

```powershell
.venv\Scripts\Activate.ps1
```

Falls eine Fehlermeldung wegen Execution Policy erscheint, einmalig ausführen:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Danach erneut aktivieren.

---

### 🐧 Wenn du Git Bash benutzt:

```bash
source .venv/Scripts/activate
```

---

Wenn es funktioniert, sollte im Terminal nun folgendes stehen:

```
(.venv)
```

---

## 4️⃣ main.py starten

Egal welches Terminal du nutzt:

```bash
python main.py
```

ODER

- `main.py` öffnen
- Oben rechts auf das ▶️ Symbol klicken („Run Python File“)

---

# 🧠 Wichtig

- Wenn das Terminal oder die `run.bat` geschlossen wird,
  wird das laufende Training sofort beendet.
