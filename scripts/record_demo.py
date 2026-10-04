"""Record an animated SVG walkthrough of the GraphRAG Streamlit demo.

Frames: multi-hop question answered (hybrid) -> graph explorer tab ->
global-mode question. Output: assets/demo.svg (kept small: single CLI
args can't exceed 128KB when pushing).
"""
import base64
import io
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.sync_api import sync_playwright  # noqa: E402

PORT = 8504
W, H = 900, 563
QUALITY = 34
FRAMES = []


def shot(page, name):
    png = page.screenshot()
    FRAMES.append((name, png))
    print("frame:", name, len(png) // 1024, "KB")


def main():
    env = dict(os.environ, PATH=os.path.dirname(sys.executable)
               + os.pathsep + os.environ["PATH"])
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.port", str(PORT),
         "--browser.gatherUsageStats", "false"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            s = socket.socket()
            try:
                s.connect(("localhost", PORT))
                s.close()
                break
            except OSError:
                time.sleep(1)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": W, "height": H})
            page.goto(f"http://localhost:{PORT}", wait_until="networkidle")
            page.get_by_role("button", name="Analyser").wait_for(timeout=90000)
            time.sleep(3)

            # frame 1: multi-hop question answered in hybrid mode
            page.get_by_role("button", name="Analyser").click()
            page.get_by_text("Réponse").first.wait_for(timeout=60000)
            time.sleep(3)
            page.get_by_text("Voir les faits utilisés").first.evaluate(
                "el => el.scrollIntoView({block: 'center'})")
            time.sleep(0.8)
            shot(page, "multihop-answer")

            # frame 2: graph explorer
            page.get_by_role("tab", name="Explorer le graphe").click()
            page.locator(".js-plotly-plot").first.wait_for(timeout=45000)
            time.sleep(2)
            page.locator(".js-plotly-plot").first.evaluate(
                "el => el.scrollIntoView({block: 'center'})")
            time.sleep(0.8)
            shot(page, "graph-explorer")

            # frame 3: global mode question
            page.get_by_role("tab", name="Poser une question").click()
            time.sleep(1)
            page.get_by_text("Global (résumés de communautés)").click()
            page.locator('[data-testid="stTextInput"] input').fill(
                "Quels sont les grands domaines d'achat de ce corpus ?")
            page.get_by_role("button", name="Analyser").click()
            page.get_by_text("Réponse").first.wait_for(timeout=60000)
            time.sleep(3)
            page.get_by_text("Voir les faits utilisés").first.evaluate(
                "el => el.scrollIntoView({block: 'center'})")
            time.sleep(0.8)
            shot(page, "global-answer")
            browser.close()
    finally:
        proc.terminate()

    from PIL import Image
    n = len(FRAMES)
    dur = n * 3
    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             'xmlns="http://www.w3.org/2000/svg" '
             'xmlns:xlink="http://www.w3.org/1999/xlink">']
    for i, (name, png) in enumerate(FRAMES):
        img = Image.open(io.BytesIO(png)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=QUALITY)
        b64 = base64.b64encode(buf.getvalue()).decode()
        start, end = i / n, (i + 1) / n
        parts.append(
            f'<image x="0" y="0" width="{W}" height="{H}" '
            f'xlink:href="data:image/jpeg;base64,{b64}" opacity="0">'
            f'<animate attributeName="opacity" values="0;1;0" '
            f'keyTimes="0;{start:.4f};{end:.4f}" calcMode="discrete" '
            f'dur="{dur}s" repeatCount="indefinite"/></image>')
    parts.append("</svg>")
    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "assets", "demo.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\n".join(parts))
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
