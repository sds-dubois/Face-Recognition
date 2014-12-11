import urllib2
import urllib
import simplejson
import os
import shutil
import Image
import imghdr

n=30
basepath = 'datapython/'

if not os.path.exists(basepath):
	os.mkdir(basepath)

print os.walk(basepath)
personalites = [name for name in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, name))]

for personalite in personalites:
	print personalite
	path = basepath+"/"+personalite
	i=1
	j=1
	for i in range(15, n):
		try:
			url = ('https://ajax.googleapis.com/ajax/services/search/images?' +
						'v=1.0&q='+personalite.replace(" ", "%20")+'&userip=129.104.222.33&start='+str(4*i))
			request = urllib2.Request(url, None, {'Referer': 'http://www.github.com'})
			response = urllib2.urlopen(request)
			results = simplejson.load(response)
			values = results['responseData']['results']
			for result in values:
				s = result['url']
				name = s[(s.rindex('/')+1):]
				fileextension = (((name.split("."))[-1]).split("%"))[0].replace("e", "").lower()
				filename = path+str(j)+"."+fileextension
				urllib.urlretrieve(s, filename)
				print(str(i)+" "+str(j))
				try:
					actualextension = imghdr.what(filename).replace("e", "")
					if fileextension != actualextension:
						print filename
						os.remove(filename)
				except Exception as e:
					print filename

				try:
					image = Image.open(filename)
				except:
					print "rotten image : "+str(j)
					os.remove(filename)
				j = j+1
		except Exception as e:
			print e
			print e.args
