
from datasource.wider import main
from datasource.context import Cache

if __name__ == '__main__':
	main(Cache('target/', (1920,1080), 0.025), 'target/mega-wipder-data/')
