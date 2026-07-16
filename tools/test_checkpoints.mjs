/**
 * E2E exercise of the checkpoint system against the dev server (port 5199):
 *  - NumericGuess: wrong answer → nudge + hint; right answer → explanation
 *  - PredictReveal: wrong pick → eliminated + retry; right pick → solved
 *  - PickLine: wrong line → eliminated; right line → solved
 *  - TryIt: resolves itself when the wired widget condition fires
 *  - meter counts up; results persist across reload
 *
 * NOTE: assertions are coupled to specific chapter content (qids, hints,
 * line numbers). If a question is reworded, update the matching regex here.
 */
import { chromium } from 'playwright-core'

const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
page.on('console', (m) => {
  if (m.type() === 'error') errors.push(m.text())
})

const fail = (msg) => {
  console.error('FAIL:', msg)
  process.exitCode = 1
}

// --- chapter 1: NumericGuess + PredictReveal -------------------------------
await page.goto('http://localhost:5199/ch/data-tokenizer', { waitUntil: 'networkidle' })

// NumericGuess: id of 'm' — wrong first (13 → "too high"), then 12
const numInput = page.getByLabel('your numeric answer').first()
await numInput.fill('13')
await page.getByRole('button', { name: 'check', exact: true }).first().click()
await page.waitForTimeout(150)
if (!(await page.getByText(/very close|too high/).first().isVisible())) fail('NumericGuess: no wrong-answer nudge')
if (!(await page.getByText(/13th letter of the alphabet/).isVisible())) fail('NumericGuess: hint not shown after miss')
await numInput.fill('12')
await page.getByRole('button', { name: 'check', exact: true }).first().click()
await page.waitForTimeout(150)
if (!(await page.getByText(/got there in 2 tries/).isVisible())) fail('NumericGuess: attempts note missing')
if (!(await page.getByText(/Position in the sorted unique-character list/).isVisible())) fail('NumericGuess: explanation missing')

// PredictReveal (digits question): wrong pick '36' → eliminated + hint; then '37'
const wrong = page.getByRole('button', { name: /^36/ })
await wrong.click()
await page.waitForTimeout(150)
if (!(await page.getByText('not that one — try again').isVisible())) fail('PredictReveal: no retry message')
if (!(await wrong.isDisabled())) fail('PredictReveal: wrong option not eliminated')
await page.getByRole('button', { name: /^37/ }).click()
await page.waitForTimeout(150)
if (!(await page.getByText(/resizes automatically from this one number/).isVisible())) fail('PredictReveal: explanation missing')

// meter should show 2 resolved
const meter = await page.getByText(/\/5 checkpoints/).textContent()
if (!meter?.startsWith('2/5')) fail(`meter shows "${meter}", expected 2/5`)

// persistence across reload
await page.reload({ waitUntil: 'networkidle' })
await page.waitForTimeout(300)
const meterAfter = await page.getByText(/\/5 checkpoints/).textContent()
if (!meterAfter?.startsWith('2/5')) fail(`after reload meter shows "${meterAfter}", expected 2/5`)
if (!(await page.getByText(/resizes automatically from this one number/).isVisible())) fail('PredictReveal: not restored after reload')

// --- chapter 5: PickLine ----------------------------------------------------
await page.goto('http://localhost:5199/ch/attention', { waitUntil: 'networkidle' })
await page.waitForTimeout(600)
const wrongLine = page.getByRole('button', { name: /^line 130:/ })
await wrongLine.click()
await page.waitForTimeout(150)
if (!(await page.getByText('not that line — look again').isVisible())) fail('PickLine: no retry message')
await page.getByRole('button', { name: /^line 121:/ }).click()
await page.waitForTimeout(150)
if (!(await page.getByText(/the entire causal structure of GPT/).isVisible())) fail('PickLine: explanation missing')

// --- chapter 2: TryIt via stepping the backward walk ------------------------
await page.goto('http://localhost:5199/ch/autograd', { waitUntil: 'networkidle' })
await page.waitForTimeout(400)
// second backward stepper (d = a·b + c²): click "step forward" until done flips
const group = page.getByRole('group', { name: 'backward stepper' }).nth(1)
const fwd = group.getByRole('button', { name: 'step forward' })
for (let i = 0; i < 12; i++) {
  if (await fwd.isDisabled()) break
  await fwd.click()
  await page.waitForTimeout(60)
}
await page.waitForTimeout(200)
if (!(await page.getByText(/identical.*walk over a graph/s).isVisible().catch(() => false))
  && !(await page.getByText(/one\s+topological sort/).isVisible().catch(() => false))) {
  fail('TryIt: backward-walk payoff not shown after finishing the walk')
}

if (errors.length) {
  console.error('CONSOLE/PAGE ERRORS:')
  for (const e of errors) console.error(' -', e)
  process.exitCode = 1
}
if (process.exitCode !== 1) console.log('ALL CHECKPOINT MECHANICS OK')
await browser.close()
