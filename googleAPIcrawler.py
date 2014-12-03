import urllib2
import urllib
import simplejson
import os
import shutil
import Image

personalites = {'barack obama', 'francois hollande', 'manuel valls'}
n=30
basepath = 'data/labeled/'

if os.path.exists(basepath):
	shutil.rmtree(basepath)
os.mkdir(basepath)

for personalite in personalites:
	print personalite
	path = basepath+personalite.replace(" ", "_")+"/"
	os.mkdir(path)
	i=1
	j=1
	for i in range(1, n):
		try:
			url = ('https://ajax.googleapis.com/ajax/services/search/images?' +
						'v=1.0&q='+personalite.replace(" ", "%20")+'&userip=INSERT-USER-IP&start='+str(4*i))
			request = urllib2.Request(url, None, {'Referer': 'http://www.google.com'})
			response = urllib2.urlopen(request)
			results = simplejson.load(response)
			values = results['responseData']['results']
			for result in values:
				s = result['url']
				name = s[(s.rindex('/')+1):]
				filename = path+str(j)+"."+(((name.split("."))[-1]).split("%"))[0]
				urllib.urlretrieve(s, filename)
				try:
					image = Image.open(filename)
				except:
					print "rotten image : "+str(j)
					os.remove(filename)
				j = j+1
		except Exception as e:
			print e
			print e.args
