
@app.post("/v1/audio/speech")
async def text_to_speech(speach_params: SpeachModel):
   pass



@app.post("/v1/audio/transcriptions")
async def audio_to_text(speach_params: SpeachModel):
    pass

