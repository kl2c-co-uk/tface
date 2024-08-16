
from datasource import Blurb, Cache, md5, ZipWalk, ensure_directory_exists, random_split, only

import datasource.config as config
from  datasource.datapoints import FacePatch, DataPoint

import datasource.datapoints as datapoints
from datasource.datapoints import split_export


def main():
	cache = Cache('target/')
	i_cartoon(cache)


def i_cartoon(cache):
	split_export(
		i_cartoon_datapoints(cache),
		8, 1,
		cache.download(
			# training = target/cb67961c4ba344c84b3e5442206436ac
			'https://drive.usercontent.google.com/download?id=1xXpE0qs2lONWKL5dqaFxqlJ_t5-glNpg&export=download&authuser=0&confirm=t&uuid=f6f6beb7-4c3b-40a7-b52d-12c62c2e84fe&at=APZUnTV9QwxtWfOsgjgqW-7icoaM:1723671279280'
		)
	)


def i_cartoon_datapoints(cache):
	annotations = cache.download(
		# annotations = target/712e3f96290bfc9c1c93a18f16ef40e8
		'https://drive.usercontent.google.com/download?id=15IHSlNBZBZs_hj6B341swc00ha5fpvB7&export=download&authuser=0&confirm=t&uuid=72fd55fe-6a76-4c73-91ee-63de54aa2775&at=APZUnTXR3ogM4tIFCGrcFoaswAor:1723670943322'
	)


	for lines in ZipWalk(annotations).text('personai_icartoonface_dettrain_anno_updatedv1.0.csv'):
		seen = []

		# start a non-datapoint
		last = ''
		data = '?unset?'

		while lines.more():
			line = lines.take()

			# get the line content
			name, l, t, r, b = line.split(',')

			# check if we need to switch datapoints
			if name != last:

				# emit the prior datapoint
				if '' != last:					
					yield DataPoint(path = last, patches = data)					
				
				# start a datapoint
				data = []
				last = name

				# check the names. maybe
				if name in seen:
					raise Exception(
						'the items are not grouped as i d expected'
					)
				seen.append(name)
			
			# add the patch tot he datapoint
			data.append(
				FacePatch(ltrb = [l, t, r, b])
			)


		
		# yield the final datapooint
		if '' != last:					
			yield DataPoint(path = last, patches = data)							



if '__main__' == __name__:
	main()
