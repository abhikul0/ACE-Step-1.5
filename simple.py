from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

import os
import torch

#check wether LM loading is enabled
init_llm_env = os.getenv("ACESTEP_INIT_LLM", "").strip().lower()

torch.mps.set_per_process_memory_fraction(0.6) # fraction of 0.75*memory

print("BEFORE")
print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

# Initialize handlers
dit_handler = AceStepHandler()
llm_handler = LLMHandler()

# Initialize services
#if using LoRa, set compile to false and remove quantization
dit_handler.initialize_service(
    project_root="/Users/abhishek/acestep-test/ACE-Step-1.5",
    config_path="acestep-v15-turbo",
    device="auto",
    use_flash_attention = False,
    compile_model = True,
    offload_to_cpu = False,
    offload_dit_to_cpu = False,
    quantization = "int8_weight_only"
)

# LoRA configuration
lora_path = ''#os.getenv("ACESTEP_LORA_PATH", "/Users/abhishek/acestep-test/ACE-Step-1.5/lora_output/final/adapter").strip()
lora_scale = float(os.getenv("ACESTEP_LORA_SCALE", "1.0"))

if lora_path:
    load_msg = dit_handler.load_lora(lora_path)
    print(f"LoRA load: {load_msg}")
    # enable LoRA
    use_lora_msg = dit_handler.set_use_lora(True)
    print(f"LoRA use: {use_lora_msg}")
    # set scale
    scale_msg = dit_handler.set_lora_scale(lora_scale)
    print(f"LoRA scale: {scale_msg}")

if init_llm_env:
    llm_handler.initialize(
        checkpoint_dir="/Users/abhishek/acestep-test/ACE-Step-1.5/checkpoints",
        lm_model_path="acestep-5Hz-lm-1.7B",
        backend="pt",
        device="cpu",
        offload_to_cpu = True,
        dtype = "torch.float16",
    )

#Configure generation parameters. refer acestep.constants.py
seed = 42
params = GenerationParams(
    vocal_language="en", #en,hi
    caption="customTag",#"A psychedelic indie rock track built on a steady, hypnotic groove from a clean bassline and a crisp drum kit. A shimmering electric guitar, treated with a subtle chorus effect, plays arpeggiated figures and chords, creating a dreamy, spacious atmosphere. The lead female vocal is breathy and ethereal, delivered with a mix of gentle intimacy and soaring intensity, often layered with harmonies in the chorus. The song progresses through verses and powerful choruses, leading to a more atmospheric bridge where the vocals become more declarative. The track concludes with an extended instrumental outro featuring melodic guitar lines and fading vocal ad-libs.",
    lyrics='''
        [Intro]
        Oh-oh-oh-oh, oh-oh-oh-oh
        Oh-oh-oh-oh, oh-oh-oh-oh
        (Check it out now)

        [Verse 1]
        The morning light is turning blue, the feeling is bizarre (bizarre)
        The night is almost over, I still don't know where you are
        The shadows, yeah, they keep me pretty like a movie star
        Daylight makes me feel like Dracula (Dracula)
        In the end, I hope it's you and me
        In the darkness, I would never leave you (ah)
        Won't ever see me in the light of day
        It's far too late, the time has come
        I'm on the verge of caving in, I run back to the dark
        Now I'm Mr. Charisma, fucking Pablo Escobar (Escobar)
        My friends are saying, "Shut up, Kevin, just get in the car" (Kevin)
        I just wanna be right where you are (oh, my love)

        [Chorus]
        In the end, I hope it's you and me
        In the darkness, I would never leave (I won't leave her)
        We both saw this moment coming from afar
        Now here we are
        Run from the sunlight, Dracula (hey)
        Run from the sunlight, Dracula (Dracu-Dracula)
        Run from the sunlight, Dracula (run from the sun)
        Isn't the view spectacular? (sunlight, Dracula)
        Oh-oh-oh-oh, oh-oh-oh-oh
        Oh-oh-oh-oh, oh-oh-oh-oh (check it out now)
        Oh-oh-oh-oh, oh-oh-oh-oh
        Oh-oh-oh-oh, oh-oh-oh-oh

        [Verse 2]
        But please, do you think about what it might mean?
        'Cause I dream about you in my sleep
        Would you ever love someone like me? Like me? (Oh)
        In the end, I hope it's you and me (oh)
        In the darkness, I would never leave (I won't leave her)
        We both saw this moment comin' from afar
        Now here we are

        [Chorus]
        So run from the sunlight, Dracula (hey)
        Run from the sunlight, Dracula (run from the sunlight, Dracula)
        Run from the sunlight, Dracula (oh)

        [Outro]
        Isn't the view spectacular? (run from the sunlight, Dracula)
        Run from the sunlight, Dracula (run from the sunlight, Dracula)
        Run from the sunlight, Dracula (run from the sunlight)
        Isn't the view spectacular? (run from the sunlight, Dracula)
        Run from the sunlight, Dracula (run from the sunlight, Dracula)
        Run from the sunlight, Dracula (run from the sunlight, Dracula)
        Isn't the view spectacular?
    ''',
    bpm=130,
    duration=200,
    keyscale="F",
    # 5Hz Language Model Parameters
    thinking = False,
    use_cot_metas = False,
    use_cot_caption = False,
    use_cot_language = False,
    seed = seed,
)

# params = GenerationParams(
#     task_type="cover",
#     src_audio="/home/abhishek/Downloads/b.mp3",
#     caption="acapella version",
#     audio_cover_strength=0.5,  # 0.0-1.0

#     # 5Hz Language Model Parameters
#     thinking = False,
#     use_cot_metas = False,
#     use_cot_caption = False,
#     use_cot_language = False,
# )

# Configure generation settings
config = GenerationConfig(
    batch_size=1,
    audio_format="mp3",
    lm_batch_chunk_size = 1,
    use_random_seed = False,
    seeds = seed
)

# Generate music
result = generate_music(dit_handler, llm_handler, params, config, save_dir="/Users/abhishek/acestep-test/ACE-Step-1.5/outputs")

print("AFTER")
print(f"Allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
print(f"Driver total: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")


# Access results
if result.success:
    for audio in result.audios:
        print(f"Generated: {audio['path']}")
        print(f"Key: {audio['key']}")
        print(f"Seed: {audio['params']['seed']}")
else:
    print(f"Error: {result.error}")