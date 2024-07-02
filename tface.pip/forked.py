
from datasource.wider import foo
from datasource.context import Cache

if __name__ == '__main__':
	foo(Cache('target/', (1920,1080), 0.2), 'target/mega-wipder-data/')
