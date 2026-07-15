/**
 * Screenshot helper for the design journal: node tools/screenshot.mjs <url-path> <out.png> [width] [height]
 */
import { chromium } from 'playwright-core'

const [, , urlPath = '/', out = 'shot.png', width = '1440', height = '900'] = process.argv
const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })
const page = await browser.newPage({ viewport: { width: Number(width), height: Number(height) } })
const errors = []
page.on('console', (m) => {
  if (m.type() !== 'error') return
  if (m.location()?.url?.endsWith('favicon.ico')) return
  errors.push(m.text())
})
page.on('pageerror', (e) => errors.push(String(e)))
await page.goto(`http://localhost:5199${urlPath}`, { waitUntil: 'networkidle' })
await page.waitForTimeout(400)
await page.screenshot({ path: out, fullPage: false })
if (errors.length) {
  console.error('CONSOLE ERRORS:')
  for (const e of errors) console.error(' -', e)
  process.exitCode = 2
} else {
  console.log('no console errors')
}
await browser.close()
