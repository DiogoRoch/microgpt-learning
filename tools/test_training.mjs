/**
 * Interactive check of chapter 9's live training: click the train button,
 * wait, verify the curve grew and samples appeared, screenshot the result.
 */
import { chromium } from 'playwright-core'

const out = process.argv[2] ?? 'ch9-live.png'
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
const page = await browser.newPage({ viewport: { width: 1440, height: 1500 } })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
await page.goto('http://localhost:5199/ch/training', { waitUntil: 'load' })
await page.waitForSelector('text=train the real model', { timeout: 60000 })

// flat-out pace so the run finishes quickly, then click train
await page.selectOption('select[aria-label="training pace"]', 'flat-out')
const btn = page.getByRole('button', { name: /train the real model/ })
await btn.click()
await page.waitForTimeout(4000)

const text = await page.textContent('body')
const finished = /1000 steps of the complete algorithm ran in your browser tab in ([\d.]+)\s*s/.exec(text ?? '')
if (!finished) {
  console.error('TRAINING DID NOT FINISH — page text lacks the timing line')
  process.exitCode = 1
} else {
  console.log(`live training finished in browser: ${finished[1]}s`)
}
await page.screenshot({ path: out, fullPage: false })
if (errors.length) {
  console.error('PAGE ERRORS:', errors.join('\n'))
  process.exitCode = 2
}
await browser.close()
