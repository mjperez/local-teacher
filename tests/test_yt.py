from youtube_transcript_api import YouTubeTranscriptApi
import traceback

def test_url(url):
    try:
        import re
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        video_id = match.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es', 'en'])
        print(f"Success for {url}: {len(transcript)} lines")
    except Exception as e:
        print(f"Error for {url}: {e}")
        traceback.print_exc()

test_url("https://www.youtube.com/watch?v=srozR_Vw97k")
test_url("https://www.youtube.com/watch?v=-iLqSkLWyow")
