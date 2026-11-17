const {chromium}=require('/tmp/codex-cosine-browser-check/node_modules/playwright');
const fs=require('fs'),http=require('http'),path=require('path'),dir=process.argv[2];
const data=JSON.parse(fs.readFileSync(path.join(dir,'data.json'))),points=data.series[0].points;
const expectedObservations=data.series.reduce((n,s)=>n+s.points.length,0);
const expectedSeries=data.series.length;
if(data.uniform_reference){
  const reference=data.series.find(s=>s.id==='uniform');
  if(!reference||reference.points.some(p=>p.y!==1/(2*p.x+1)))throw Error('Incorrect uniform share');
}
const cursorLayer=points[0].x+.225*(points[points.length-1].x-points[0].x);
const pinnedLayer=points.reduce((a,b)=>Math.abs(b.x-cursorLayer)<Math.abs(a.x-cursorLayer)?b:a).x;
(async()=>{
  const server=http.createServer((req,res)=>{res.setHeader('Content-Type','text/html');res.end(fs.readFileSync(path.join(dir,'preview.html')))});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));let browser;
  try{
    browser=await chromium.launch({executablePath:'/snap/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    const page=await browser.newPage({viewport:{width:768,height:800},colorScheme:'dark'}),errors=[],checks=[];
    page.on('pageerror',e=>errors.push(String(e)));await page.goto('http://127.0.0.1:'+server.address().port);
    const frame=page.frameLocator('iframe');await frame.locator('[data-mean-line]').first().waitFor();
    for(const width of [768,392,352]){
      await page.setViewportSize({width,height:800});await page.waitForTimeout(250);
      const result=await frame.locator('#embedding-v-energy-lines').evaluate(root=>{
        const svg=root.querySelector('svg'),bounds=svg.getBoundingClientRect(),labels=[...svg.querySelectorAll('text')],issues=[];
        labels.forEach((a,i)=>{const x=a.getBoundingClientRect();if(x.left<bounds.left-1||x.right>bounds.right+1||x.top<bounds.top-1||x.bottom>bounds.bottom+1)issues.push('overflow');
          labels.slice(i+1).forEach(b=>{const y=b.getBoundingClientRect();if(x.left<y.right+4&&x.right+4>y.left&&x.top<y.bottom+4&&x.bottom+4>y.top)issues.push('overlap')});});
        const box=svg.querySelector('[data-chart-frame]').getBoundingClientRect();
        for(const line of svg.querySelectorAll('[data-mean-line]')){const b=line.getBoundingClientRect();if(b.left<box.left||b.right>box.right||b.top<box.top||b.bottom>box.bottom)issues.push('line outside');}
        return {width:root.clientWidth,lines:svg.querySelectorAll('[data-mean-line]').length,observations:svg.querySelectorAll('[data-observation]').length,axes:svg.querySelectorAll('.axis-title').length,issues,overflow:root.scrollWidth>root.clientWidth+1};
      });
      if(width===768){
        const hit=await frame.locator('[data-chart-hit]').boundingBox(),px=hit.x+6+.225*(hit.width-12);
        await page.mouse.move(px,hit.y+30);result.hover=await frame.locator('.tooltip').innerText();
        result.rows=await frame.locator('[data-tooltip-series]').count();
        const iframe=await page.locator('iframe').boundingBox();result.guideError=await frame.locator('[data-chart-hover-guide]').evaluate((g,expected)=>Math.abs(g.getBoundingClientRect().left-expected),px-iframe.x);
        await page.mouse.click(px,hit.y+30);result.pinned=await frame.locator('.tooltip').innerText();
        await frame.locator('button[data-series="no_embed_norm"]').click();result.hidden=await frame.locator('[data-series-group="no_embed_norm"]').evaluate(g=>g.getAttribute('display')==='none');result.toggleRows=await frame.locator('[data-tooltip-series]').count();
        await frame.locator('button[data-series="no_embed_norm"]').click();await page.mouse.click(px,hit.y+30);await page.mouse.move(0,0);
      }
      await frame.locator('#embedding-v-energy-lines').screenshot({path:path.join(dir,`chart-${width}.png`)});checks.push(result);
    }
    fs.writeFileSync(path.join(dir,'browser_checks.json'),JSON.stringify({checks,errors},null,2));console.log(JSON.stringify({checks,errors}));
    if(errors.length||checks.some(c=>c.lines!==expectedSeries||c.observations!==expectedObservations||c.axes!==2||c.issues.length||c.overflow)||checks[0].rows!==expectedSeries||checks[0].guideError>1||!checks[0].hidden||checks[0].toggleRows!==expectedSeries-1||!checks[0].pinned.startsWith('Layer '+pinnedLayer+'\n'))throw Error('Chart check failed');
  }finally{if(browser)await browser.close();server.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
