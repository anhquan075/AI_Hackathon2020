var Scraper = require('images-scraper');
 
const google = new Scraper({
  puppeteer: {
    headless: false,
  }
});
var fs = require('fs'); 

(async () => {
  const results = await google.scrape('cây đổ', 400);
  const data = JSON.stringify(results);
  fs.writeFile('cay_do.json', data, (err) => {
    if (err) throw err;
    console.log('Data written to file');
  });
})();