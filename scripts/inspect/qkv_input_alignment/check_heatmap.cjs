// Validate the six-layer alignment heatmap and hover/tap at desktop/mobile widths.
const {chromium}=require('/tmp/codex-cosine-browser-check/node_modules/playwright');
const fs=require('fs'),http=require('http'),path=require('path');
const dir=process.argv[2];
const data=JSON.parse(fs.readFileSync(path.join(dir,'data.json')));
const expectedCells=data.panels.reduce((n,p)=>n+p.mean.length,0);
(async()=>{
  const server=http.createServer((req,res)=>{res.setHeader('Content-Type','text/html');res.end(fs.readFileSync(path.join(dir,'preview.html')))});
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try{
    browser=await chromium.launch({executablePath:'/snap/bin/chromium',headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    const page=await browser.newPage({viewport:{width:1056,height:1100},colorScheme:'light'}),errors=[],checks=[];
    page.on('pageerror',e=>errors.push(String(e)));
    await page.goto('http://127.0.0.1:'+server.address().port);
    const frame=page.frameLocator('iframe');
    await frame.locator('[data-alignment-cell]').first().waitFor({timeout:30000});
    for(const width of [1056,768,392,352]){
      await page.setViewportSize({width,height:1100});await page.waitForTimeout(300);
      const result=await frame.locator('body').evaluate(body=>{
        const roots=[...body.querySelectorAll('#qkv-alignment-heatmap')];
        const problems=[];
        for(const svg of body.querySelectorAll('.heatmap-chart')){
          const bound=svg.getBoundingClientRect(),labels=[...svg.querySelectorAll('text')];
          for(let i=0;i<labels.length;i++){
            const a=labels[i].getBoundingClientRect();
            if(a.left<bound.left-1||a.right>bound.right+1||a.top<bound.top-1||a.bottom>bound.bottom+1)problems.push('overflow '+labels[i].textContent);
            for(let j=i+1;j<labels.length;j++){
              const b=labels[j].getBoundingClientRect();
              if(a.left<b.right+4&&a.right+4>b.left&&a.top<b.bottom+4&&a.bottom+4>b.top)problems.push('overlap '+labels[i].textContent+'/'+labels[j].textContent);
            }
          }
          const f=svg.querySelector('[data-chart-frame]').getBoundingClientRect();
          for(const cell of svg.querySelectorAll('[data-alignment-cell]')){
            const c=cell.getBoundingClientRect();if(c.left<f.left-1||c.right>f.right+1||c.top<f.top-1||c.bottom>f.bottom+1)problems.push('cell outside frame');
          }
        }
        return {width:roots[0].clientWidth,charts:body.querySelectorAll('.heatmap-chart').length,
          cells:body.querySelectorAll('[data-alignment-cell]').length,axes:body.querySelectorAll('.axis-title').length,
          colorMax:roots.map(r=>r.querySelector('.color-max').textContent),problems,
          horizontalOverflow:body.scrollWidth>body.clientWidth+1};
      });
      if(width===1056){
        const hit=await frame.locator('#qkv-alignment-heatmap [data-chart-hit]').first().boundingBox();
        await page.mouse.move(hit.x+2,hit.y+2);
        result.hover=await frame.locator('#qkv-alignment-heatmap .heatmap-tooltip').innerText();
        await page.mouse.click(hit.x+2,hit.y+2);
        result.pinned=await frame.locator('#qkv-alignment-heatmap .heatmap-detail').innerText();
        if(!result.hover.includes('Emb')||!result.pinned.includes('262,144 valid tokens')||!result.pinned.includes(' · k '))throw Error('Hover/pin incorrect');
        await page.mouse.move(0,0);
      }
      await frame.locator('#qkv-alignment-heatmap').screenshot({path:path.join(dir,`variant-${width}.png`)});
      checks.push(result);
    }
    const result={checks,errors};fs.writeFileSync(path.join(dir,'browser_checks.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
    if(errors.length||checks.some(c=>c.charts!==6||c.cells!==expectedCells||c.axes!==12||c.colorMax.some(v=>v!==c.colorMax[0])||c.problems.length||c.horizontalOverflow))throw Error('Browser validation failed');
  }finally{if(browser)await browser.close();server.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
