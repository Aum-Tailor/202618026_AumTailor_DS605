import scrapy

class Book(scrapy.Spider):
    name = "book_scraper"
    start_urls = ["https://books.toscrape.com/catalogue/page-1.html"]
    
    pages_scraped = 1
    
    star_mapping = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    
    def parse(self, response):
        book_hrefs = response.css("article.product_pod h3 a::attr(href)").getall()
        for href in book_hrefs:
            yield response.follow(href, self.extract_book_info)
            
        if self.pages_scraped < 5:
            next_page_link = response.css("li.next a::attr(href)").get()
            if next_page_link:
                self.pages_scraped += 1
                yield response.follow(next_page_link, self.parse)
                
    def extract_book_info(self, response):
        rating_class = response.css("p.star-rating::attr(class)").get()
        rating_text = rating_class.split()[-1] if rating_class else ""
        rating_num = self.star_mapping.get(rating_text)
        
        avail_raw = response.css("p.instock.availability::text").getall()
        avail_clean = "".join(avail_raw).replace('\n', '').strip()
        
        table_dict = {}
        for row in response.css("table.table-striped tr"):
            th = row.css("th::text").get()
            td = row.css("td::text").get()
            if th and td:
                table_dict[th] = td
                
        yield {
            "title": response.css("div.product_main h1::text").get(),
            "category": response.xpath('//ul[@class="breadcrumb"]/li[3]/a/text()').get(),
            "price": response.css("p.price_color::text").get(),
            "rating": rating_num,
            "availability": avail_clean,
            "description": response.css("#product_description ~ p::text").get(),
            "upc": table_dict.get("UPC"),
            "number_of_reviews": table_dict.get("Number of reviews"),
            "product_url": response.url,
        }