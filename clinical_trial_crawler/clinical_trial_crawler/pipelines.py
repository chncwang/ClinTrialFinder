# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


import json
from typing import Any, Dict, List
from scrapy import Item

# useful for handling different item types with a single interface
class JsonWriterPipeline:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def process_item(self, item: Item, spider: Any) -> Item:
        self.items.append(dict(item))
        if hasattr(spider, "output_file") and spider.output_file:
            # Write immediately for single-trial fetches
            with open(spider.output_file, "w") as f:
                json.dump([dict(item)], f)
        return item

    def close_spider(self, spider: Any) -> None:
        if not hasattr(spider, "output_file") or not spider.output_file:
            # Only write to default output if no specific file is specified
            with open("items.json", "w") as f:
                json.dump(self.items, f)
