import base64
import requests
import json
APIURL="https://rest.rchilli.com/RChilliParser/Rchilli/parseResumeBinary"
USERKEY = 'Your UserKey'
VERSION = '8.0.0'
subUserId = 'Your Company Name'

# file absolutepath
filePath='/home/resumefolder/SampleResume.docx'
fileName='SampleResume.docx'
# service url- provided by RChilli


with open(filePath, "rb") as filePath:
    encoded_string = base64.b64encode(filePath.read())
data64 = encoded_string.decode('UTF-8')

headers = {'content-type': 'application/json'}

body =  """{"filedata":\""""+data64+"""\","filename":\""""+ fileName+"""\","userkey":\""""+ USERKEY+"""\",\"version\":\""""+VERSION+"""\",\"subuserid\":\""""+subUserId+"""\"}"""

response = requests.post(APIURL,data=body,headers=headers)
resp =json.loads(response.text)
#please handle error too
Resume =resp["ResumeParserData"]
#read values from response
print (Resume["Name"]["FirstName"])
print (Resume["Name"]["LastName"])
print (Resume["Email"])
print (Resume["SegregatedExperience"])


