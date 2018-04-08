# save your ckey, csecret, atoken, asecret details in twitterdetails and then import that python file
from twitterdetails import *
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

class listener(StreamListener):
    def on_data(self, data):
        try:
            all_data = json.loads(data)
            tweet = all_data['text']
            username = all_data['user']['screen_name']
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, sentiment_value, confidence)

            if confidence >= 80:
                f = open('twitter-out.txt', 'a')
                f.write(sentiment_value)
                f.write('\n')
                f.close()
            return True
        except:
            return True

    def on_error(self, status_code):
        print(status_code)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
# enter the topic of your interest
twitterStream.filter(track=['USA'])