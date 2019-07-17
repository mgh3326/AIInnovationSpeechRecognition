#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[2]:


import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./TEST-369d71cc40d2.json"

# In[3]:


# In[4]:


import argparse
import io


# [START tts_synthesize_text]
def synthesize_text(text, out_file):
  """Synthesizes speech from the input string of text."""
  from google.cloud import texttospeech
  client = texttospeech.TextToSpeechClient()

  input_text = texttospeech.types.SynthesisInput(text=text)

  # Note: the voice can also be specified by name.
  # Names of voices can be retrieved with client.list_voices().
  voice = texttospeech.types.VoiceSelectionParams(
    language_code='ko-KR',
    ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE,
    name='ko-KR-Wavenet-A')

  audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
    speaking_rate=1.0,
    sample_rate_hertz=16000,
    volume_gain_db=0.0)

  response = client.synthesize_speech(input_text, voice, audio_config)

  # The response's audio_content is binary.
  with open(out_file, 'wb') as out:
    out.write(response.audio_content)
    print('Audio content written to file %s' % out_file)


# [END tts_synthesize_text]


# [START tts_synthesize_ssml]
def synthesize_ssml(ssml):
  """Synthesizes speech from the input string of ssml.

  Note: ssml must be well-formed according to:
      https://www.w3.org/TR/speech-synthesis/

  Example: <speak>Hello there.</speak>
  """
  from google.cloud import texttospeech
  client = texttospeech.TextToSpeechClient()

  input_text = texttospeech.types.SynthesisInput(ssml=ssml)

  # Note: the voice can also be specified by name.
  # Names of voices can be retrieved with client.list_voices().
  voice = texttospeech.types.VoiceSelectionParams(
    language_code='en-US',
    ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)

  audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.MP3)

  response = client.synthesize_speech(input_text, voice, audio_config)

  # The response's audio_content is binary.
  with open('output.mp3', 'wb') as out:
    out.write(response.audio_content)
    print('Audio content written to file "output.mp3"')


# [END tts_synthesize_ssml]


# In[5]:


synthesize_text("지금은 공덕에서 지금 음성인식 교육중입니다", "./tts.wav")

# In[6]:


filename = "./tts.wav"

# In[ ]:
