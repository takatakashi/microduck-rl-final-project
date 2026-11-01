"""Build review videos from verified recordings, without changing simulation time.

Run on macOS from the project root: python3 docs/video/build_review.py
Requires ffmpeg/ffprobe and a Japanese system font. Does not import training code.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'tmp/video-build'
OUT = ROOT / 'output/video'
REVIEW = ROOT / 'outputs/review'
FONT = next(p for p in Path('/System/Library/Fonts').iterdir()
            if unicodedata.normalize('NFC',p.name) == 'ヒラギノ角ゴシック W4.ttc')
FONT = str(FONT).replace('\\','\\\\').replace(':','\\:').replace("'", "'\\''")
counter = 0
manifest = {'purpose':'review; not uploaded or submitted', 'playback_speed':1,
            'all_trials_crop':[0,160,640,280],
            'all_trials_edit':'All original frames and captions retained; crop removes scene sky/floor only. Result cards mark episode boundaries.',
            'comparison_edit':'Same seed and initial root state, simultaneous start, complete clips at 1x. Shorter clip freezes after its end with an explicit label.',
            'inputs':[], 'outputs':[]}


def run(args):
    subprocess.run(args,check=True,capture_output=True,text=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    r=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],
                     check=True,capture_output=True,text=True)
    return json.loads(r.stdout)


def caption(text,x,y,size=30,color='white',enable=None):
    global counter
    counter+=1
    path=WORK/f'text_{counter:03d}.txt'
    path.write_text(text,encoding='utf-8')
    relative=path.relative_to(ROOT).as_posix()
    filt=f"drawtext=fontfile='{FONT}':textfile='{relative}':expansion=none:fontcolor={color}:fontsize={size}:x={x}:y={y}"
    if enable:filt+=f":enable='{enable}'"
    return filt


def encode(inputs,filters,path,duration=None):
    graph=path.with_suffix('.filter.txt')
    graph.write_text(filters,encoding='utf-8')
    args=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',*inputs,
          '-filter_complex_script',str(graph),'-map','[out]','-an']
    if duration is not None:args+=['-t',str(duration)]
    run(args+['-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p',
              '-r','50','-movflags','+faststart','-map_metadata','-1',str(path)])


def card(name,lines,seconds):
    path=WORK/(name+'.mp4')
    filters=['drawbox=x=60:y=78:w=1160:h=5:color=0x23cbb0:t=fill']
    for text,y,size,color in lines:filters.append(caption(text,60,y,size,color))
    encode(['-f','lavfi','-i','color=c=0x0d1c2a:s=1280x720:r=50'],
           '[0:v]'+','.join(filters)+'[out]',path,seconds)
    return path


def checked_input(folder,episode):
    path=folder/episode['video']['file']
    assert digest(path)==episode['video']['sha256'],path
    data=probe(path);stream=next(s for s in data['streams'] if s['codec_type']=='video')
    assert (stream['width'],stream['height'],stream['r_frame_rate'])==(640,480,'50/1')
    duration=float(data['format']['duration'])
    manifest['inputs'].append({'path':path.relative_to(ROOT).as_posix(),
                               'sha256':digest(path),'duration_seconds':duration})
    return path,duration


def concatenate(parts,name):
    file=WORK/(name+'.concat.txt')
    file.write_text(''.join("file '"+str(p).replace("'", "'\\''")+"'\n" for p in parts))
    out=OUT/(name+'.mp4')
    run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0',
         '-i',str(file),'-c','copy','-movflags','+faststart','-map_metadata','-1',str(out)])
    run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(out),'-f','null','-'])
    data=probe(out)
    actual=float(data['format']['duration'])
    expected=sum(float(probe(p)['format']['duration']) for p in parts)
    assert abs(actual-expected)<.1,(actual,expected)
    timeline=[];start=0.
    for p in parts:
        d=float(probe(p)['format']['duration'])
        timeline.append({'segment':p.stem,'start_seconds':round(start,3),'duration_seconds':d})
        start+=d
    manifest['outputs'].append({'file':out.name,'duration_seconds':actual,'sha256':digest(out),
                                'decode':'passed','timeline':timeline})
    print(f'{out.name}: {actual:.2f}s',flush=True)


def build_all_trials():
    folder=REVIEW/'hybrid_goal_fast_validation'
    r=json.loads((folder/'evaluation.json').read_text())
    assert r['status']=='completed' and r['seeds']==list(range(500,510))
    assert all(e['success'] for e in r['episodes']) and len(r['episodes'])==10
    parts=[card('all_intro',[
        ('MICRODUCK  |  動作確認用',112,27,'0x23cbb0'),
        ('目標へ歩いて、2秒間停止する',196,48,'white'),
        ('自分でPPO学習したモデル ＋ ルールによる目標到達制御',298,30,'white'),
        ('速度変更版 / 未使用seed 500〜509 / 全10試行・等速',368,30,'white'),
        ('上限0.4 m/s・目標半径15 cm・2秒停止の基準を固定',442,28,'0xb6c8d6'),
        ('上下の余白をトリミングして拡大。各試行を最初から最後まで収録。',558,25,'0xb6c8d6')],6)]
    for i,e in enumerate(r['episodes'],1):
        src,duration=checked_input(folder,e)
        out=WORK/f'all_seed_{e["seed"]}.mp4'
        filters=('[0:v]split[cap][scene];'
                 '[cap]crop=640:48:0:0,scale=1280:96[header];'
                 '[scene]crop=640:280:0:160,scale=1280:560[body];'
                 '[header][body]vstack=inputs=2,pad=1280:720:0:64:color=0x0d1c2a,'+
                 caption(f'速度変更版  |  試行 {i}/10  |  seed {e["seed"]}  |  等速・全編',30,17,29)+'[out]')
        encode(['-i',str(src)],filters,out)
        assert abs(float(probe(out)['format']['duration'])-duration)<.001
        parts.append(out)
        parts.append(card(f'all_result_{e["seed"]}',[
            (f'試行 {i}/10  |  seed {e["seed"]}',140,30,'0x23cbb0'),
            ('移動・停止に成功',244,50,'white'),
            (f'停止完了 {e["duration_seconds"]:.2f}秒  /  目標誤差 {e["arrival_error_m"]*100:.1f} cm',352,34,'white'),
            ('距離・移動速度・旋回速度の基準を2秒維持',440,28,'0xb6c8d6'),
            ('試行の区切り：次の試行では初期状態をリセット' if i<10 else '全10試行終了。続いて集計結果を表示します。',554,26,'0xb6c8d6')],2))
        print(f'all trials: {i}/10 encoded',flush=True)
    s=r['summary']
    parts.append(card('all_summary',[
        ('全10試行の結果  |  seed 500〜509',112,29,'0x23cbb0'),
        (f'成功 {s["successes"]}/{s["episodes"]}  /  転倒 0',205,53,'white'),
        (f'停止完了まで平均 {s["mean_duration_seconds"]:.2f}秒  /  誤差平均 {s["mean_arrival_error_m"]*100:.1f} cm',316,34,'white'),
        ('最短5.64秒・最長22.86秒。最長の試行も省略せず収録。',408,29,'white'),
        ('速度比較は別動画の同じseed・初期状態の3試行を参照。',486,27,'0xb6c8d6'),
        ('機体・学習基盤: Pollen Robotics / microduck_rl, mjlab, rsl_rl',584,23,'0xb6c8d6')],8))
    concatenate(parts,'microduck-fast-validation-all')


def build_comparison():
    old_dir=REVIEW/'hybrid_goal_pilot';fast_dir=REVIEW/'hybrid_goal_fast_pilot'
    old=json.loads((old_dir/'evaluation.json').read_text());fast=json.loads((fast_dir/'evaluation.json').read_text())
    parts=[card('compare_intro',[
        ('MICRODUCK  |  速度変更前後の比較',112,29,'0x23cbb0'),
        ('左：変更前     右：速度変更版',205,48,'white'),
        ('同じseed・同じ初期位置と姿勢 / 開発用の3試行',316,32,'white'),
        ('左右を同時に開始し、どちらも等速で再生',404,32,'white'),
        ('先に終了した側は、終了フレームを静止して表示します。',510,27,'0xb6c8d6')],5)]
    for o,e in zip(old['episodes'],fast['episodes']):
        assert o['seed']==e['seed'] and o['initial_state']==e['initial_state']
        left,ld=checked_input(old_dir,o);right,rd=checked_input(fast_dir,e)
        duration=max(ld,rd)
        out=WORK/f'compare_seed_{e["seed"]}.mp4'
        filters=(f'[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={round((duration-ld)*50)}[a];'
                 f'[1:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={round((duration-rd)*50)}[b];'
                 '[a][b]hstack=inputs=2,pad=1280:720:0:118:color=0x0d1c2a,'+
                 ','.join([
                     caption(f'変更前  |  seed {e["seed"]}',28,29,30),
                     caption(f'速度変更版  |  seed {e["seed"]}',668,29,30,'0x23cbb0'),
                     caption(f'停止完了 {o["duration_seconds"]:.2f}秒',28,77,25),
                     caption(f'停止完了 {e["duration_seconds"]:.2f}秒',668,77,25),
                     caption('成功 / 終了フレームで静止',28,625,25,'0x23cbb0',f'gte(t,{ld})'),
                     caption('成功 / 終了フレームで静止',668,625,25,'0x23cbb0',f'gte(t,{rd})'),
                     caption('同じ初期状態から同時開始 / 等速 / 次のseedへの切替時はリセット',28,679,23,'0xb6c8d6')])+'[out]')
        encode(['-i',str(left),'-i',str(right)],filters,out,duration)
        assert abs(float(probe(out)['format']['duration'])-duration)<.001
        parts.append(out)
        print(f'comparison: seed {e["seed"]} encoded',flush=True)
    parts.append(card('compare_summary',[
        ('共通の開発3試行での実測',112,29,'0x23cbb0'),
        ('停止完了時間  9.31秒 → 8.01秒',210,46,'white'),
        ('約14%短縮 / 成功は両方3/3 / 転倒は両方0',310,32,'white'),
        ('前進中の平均実速度  0.139 → 0.188 m/s',406,34,'white'),
        ('変更は接近速度の距離係数のみ。追加学習なし。',494,28,'0xb6c8d6'),
        ('異なるseedの10試行平均を、同条件の速度比較には使っていません。',585,24,'0xb6c8d6')],6))
    concatenate(parts,'microduck-speed-comparison')


if __name__=='__main__':
    import os
    os.chdir(ROOT)
    WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    build_all_trials()
    build_comparison()
    manifest['builder_sha256']=digest(Path(__file__))
    (OUT/'review_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
