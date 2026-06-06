# Shipping Singularity to your friend

The experience you wanted: he gets one file, double-clicks it, a window opens on
his computer, he signs up / logs in, a setup checklist tells him exactly what to
do and tracks his progress, then he presses Start. No code, ever.

## The one unavoidable extra: IB Gateway

Interactive Brokers' API only talks to their own gateway program running on the
same machine — no trading app of any kind can skip this; it's IBKR's design.
The in-app checklist walks him through it. So he installs exactly two things:
**IB Gateway** (free) and **the Singularity app** you send. Nothing else — no
Python, no editor.

## Step 1 — You build the app (one command, on your friend's OS)

PyInstaller makes a native app for whatever OS you build on, and cannot
cross-build:
- Friend on **Windows** → build on Windows → you get `dist\Singularity.exe`
- Friend on **Mac** → build on a Mac → you get `dist/Singularity`

```bash
python build.py
```

No machine of his OS? A free GitHub Actions runner (windows-latest /
macos-latest) builds it for you.

## Step 2 — You send the one file

- **Windows:** zip and send `dist\Singularity.exe`. SmartScreen may warn about an
  unknown publisher → *More info → Run anyway*. (A code-signing cert removes it.)
- **Mac:** zip and send `dist/Singularity`. Gatekeeper warns once → right-click →
  *Open*. (Notarizing removes it; needs an Apple Developer account.)

## Step 3 — He runs it (no code)

1. Double-click **Singularity** → a window opens on his computer.
2. **Sign up** (first time) or **log in**. The account is stored locally and just
   keeps the app private to him.
3. The **setup checklist** appears as a notification at the top and walks him
   step by step:
   install Gateway → enable the API → log into IB (Paper first) →
   **Test connection** (the app verifies this itself and ticks it green) →
   pick symbol + risk:reward → run in Paper.
   The progress bar fills as he finishes; it remembers what's done between runs.
4. When the bar is full he presses **Start** and watches the equity curve, the
   per-trade P&L, and the "what it has learned" panel.

Closing the window stops the bot. *Stop & flatten* closes open positions.

## One honest line to pass along

The app is real, working software, but no tool — this one included — can
guarantee profit; if such a thing existed it wouldn't be handed out. Keep it in
**Paper** until the equity curve and win rate hold up over many trades, and only
risk money you can afford to lose.
