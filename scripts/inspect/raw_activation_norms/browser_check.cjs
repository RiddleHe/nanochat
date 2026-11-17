// Usage: node browser_check.cjs /absolute/visualization/archive
const {chromium}=require('/tmp/codex-cosine-browser-check/node_modules/playwright');
const fs=require('fs'),http=require('http'),path=require('path');
const dir=process.argv[2];
const data=JSON.parse(fs.readFileSync(path.join(dir,'data.json')));
const attentionOnly=data.families==='attention',expectedSeries=data.series?data.series.length:attentionOnly?1:2;
const toggleId=data.comparison?'variant':'mlp',hasToggle=expectedSeries>1;
(async()=>{
  const server=http.createServer((req,res)=>{res.setHeader('Content-Type','text/html');res.end(fs.readFileSync(path.join(dir,'preview.html')))});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    browser=await chromium.launch({executablePath:'/snap/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    const page=await browser.newPage({viewport:{width:768,height:800},colorScheme:'dark'});
    const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('http://127.0.0.1:'+server.address().port);
    const frame=page.frameLocator('iframe');
    await frame.locator('[data-mean-line]').first().waitFor({timeout:30000});
    const checks=[];
    for(const width of [768,392,352]){
      await page.setViewportSize({width,height:850});await page.waitForTimeout(250);
      const result=await frame.locator('#full-validation-raw-norms').evaluate(root=>{
        const svg=root.querySelector('svg'),bounds=svg.getBoundingClientRect(),labels=[...svg.querySelectorAll('text')];
        const overlap=[];
        labels.forEach((a,i)=>labels.slice(i+1).forEach(b=>{const x=a.getBoundingClientRect(),y=b.getBoundingClientRect();if(x.left<y.right+4&&x.right+4>y.left&&x.top<y.bottom+4&&x.bottom+4>y.top)overlap.push([a.textContent,b.textContent])}));
        const overflow=labels.filter(l=>{const x=l.getBoundingClientRect();return x.left<bounds.left-1||x.right>bounds.right+1||x.top<bounds.top-1||x.bottom>bounds.bottom+1}).map(l=>l.textContent);
        const frame=svg.querySelector('[data-chart-frame]').getBoundingClientRect();
        const references=[...svg.querySelectorAll('[data-reference]')].map(line=>({id:line.dataset.reference,value:Number(line.dataset.value),horizontal:line.getAttribute('y1')===line.getAttribute('y2'),dashed:!!line.getAttribute('stroke-dasharray'),inside:line.getBoundingClientRect().top>=frame.top&&line.getBoundingClientRect().bottom<=frame.bottom}));
        const outOfFrame=[...svg.querySelectorAll('[data-mean-line],[data-band]')].filter(p=>{const b=p.getBoundingClientRect();return b.left<frame.left-1||b.right>frame.right+1||b.top<frame.top-1||b.bottom>frame.bottom+1}).length;
        return {width:root.clientWidth,lines:svg.querySelectorAll('[data-mean-line]').length,bands:svg.querySelectorAll('[data-band]').length,
          observations:[...svg.querySelectorAll('[data-mean-line]')].reduce((n,p)=>n+p.__data__.length,0),axes:svg.querySelectorAll('.axis-title').length,
          references,overlap,overflow,outOfFrame,horizontalOverflow:root.scrollWidth>root.clientWidth+1};
      });
      if(width===768){
        const hit=await frame.locator('[data-chart-hit]').boundingBox();
        const px=hit.x+6+(12.25+1)/24*(hit.width-12);
        await page.mouse.move(px,hit.y+50);
        result.hover=await frame.locator('.tooltip').innerText();
        result.hoverRows=await frame.locator('[data-tooltip-series]').count();
        const iframeBox=await page.locator('iframe').boundingBox();
        result.guideError=await frame.locator('[data-chart-hover-guide]').evaluate((guide,expected)=>Math.abs(guide.getBoundingClientRect().left-expected),px-iframeBox.x);
        await page.mouse.click(px,hit.y+50);
        result.pinned=await frame.locator('.tooltip').innerText();
        if(hasToggle)await frame.locator(`button[data-series="${toggleId}"]`).click();
        result.toggleHidesMlp=hasToggle?await frame.locator(`[data-series-group="${toggleId}"]`).evaluate(g=>g.getAttribute('display')==='none'):(await frame.locator('[data-series-group="mlp"]').count()===0);
        result.toggleRows=await frame.locator('[data-tooltip-series]').count();
        await page.waitForTimeout(250);
        await frame.locator('#full-validation-raw-norms').screenshot({path:path.join(dir,'attention-only.png')});
        if(hasToggle)await frame.locator(`button[data-series="${toggleId}"]`).click();
        await page.mouse.click(px,hit.y+50);await page.mouse.move(0,0);
        await page.waitForTimeout(250);
      }
      await frame.locator('#full-validation-raw-norms').screenshot({path:path.join(dir,`chart-${width}.png`)});
      checks.push(result);
    }
    const result={checks,errors};fs.writeFileSync(path.join(dir,'browser_checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
    if(checks.some(c=>c.references.length!==(data.references||[]).length||c.references.some((r,i)=>r.value!==data.references[i].value||!r.horizontal||!r.dashed||!r.inside)))throw Error('Reference validation failed');
    if(errors.length||checks.some(c=>c.lines!==expectedSeries||c.bands!==expectedSeries||c.observations!==data.points.length||c.axes!==2||c.overlap.length||c.overflow.length||c.outOfFrame||c.horizontalOverflow)||checks[0].hoverRows!==expectedSeries||checks[0].guideError>1||!checks[0].toggleHidesMlp||checks[0].toggleRows!==1||!checks[0].pinned.startsWith('L12'))throw Error('Browser validation failed');
  }finally{if(browser)await browser.close();server.close()}
})().catch(error=>{console.error(error);process.exitCode=1});
