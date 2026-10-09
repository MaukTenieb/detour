#!/usr/bin/env python3
"""Detour: cut a sheet of eight portraits (the same figure, eight expressions) into a clean sprite strip.

Three judges look at every panel:
  1. the backdrop judge: a smooth model of the painted backdrop, removed where it is reached from the panel's edges;
  2. two neural judges (U2-Net, IS-Net), run offline through ONNX.
Where they agree, it is settled; where they disagree, the majority decides, and the other seven panels, aligned on
this one, settle what is left. Edges get a real alpha (foreground and backdrop colours estimated locally) and their
colour is freed from the backdrop's. Every frame is scaled on the head and shoulders, top of the head on one line.
The program checks its own work: a full-size proof on two backgrounds and a report of whatever it doubts."""
import argparse,hashlib,json,os,sys,urllib.request
import cv2,numpy as np

MODELS={'u2net':320,'isnet-general-use':1024}
MODEL_URL='https://github.com/danielgatis/rembg/releases/download/v0.0.0/{}.onnx'
CACHE=os.path.expanduser(os.environ.get('DETOUR_CACHE','~/.cache/detour'))

# ---------------------------------------------------------------- panels
def lab(im):return cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(np.float32)
def _lines(L,axis,q=50):
    d=np.abs(np.diff(L,axis=axis)).sum(-1)
    return np.percentile(d,q,axis=0) if axis==1 else np.percentile(d,q,axis=1)
def _cuts(prof,n,size):
    out=[]
    for k in range(1,n):
        c=size*k/n;lo,hi=int(c-size*.08),int(c+size*.08);w=prof[lo:hi];base=np.median(w)
        sig=np.where(w>max(6,4*base))[0]
        if len(sig):
            s=sig[np.argmax(w[sig])];cl=sig[np.abs(sig-s)<=30];a,b=lo+cl.min(),lo+cl.max()+1
            if b-a<=1:a=b=lo+s+1
        else:a=b=int(round(c))
        out.append((int(a),int(b)))
    bd=[0]+[x for ab in out for x in ab]+[size];return [(bd[2*i],bd[2*i+1]) for i in range(n)]
def _trim(L,box):
    """step inside drawn frames and gutter remnants: an edge strip, uniform along its length, ended by an unbroken line"""
    (y0,y1),(x0,x1)=box;P=L[y0:y1,x0:x1];h,w=P.shape[:2]
    def side(Q,lim):
        d=np.percentile(np.abs(np.diff(Q,axis=0)).sum(-1),20,axis=1);base=np.median(d);flat=Q.std(1).mean(-1);best=0
        for k in np.where(d[:lim]>max(6,4*base))[0]:
            if np.all(flat[:k+1]<10):best=k+1
        return best
    m=.045;T=P.transpose(1,0,2)
    t=side(P,int(h*m));b=side(P[::-1],int(h*m));l=side(T,int(w*m));r=side(T[::-1],int(w*m))
    return (y0+t,y1-b),(x0+l,x1-r)
def _table_lines(P):
    """candidate table tops: a horizontal edge crossing most of the panel low down, over a band that is flat from side to side"""
    h,w=P.shape[:2];d=np.abs(P[2:]-P[:-2]).sum(-1);frac=(d>20).mean(1);out=[];last=-9
    for y in range(int(h*.72),h-2):
        if frac[y-1]>.5 and y-last>3:
            gx=np.abs(np.diff(P[y+2:h],axis=1)).sum(-1);out.append((y,float((gx<6).mean())))
        if frac[y-1]>.5:last=y
    return out
def find_panels(im,rows=2,cols=4):
    L=lab(im);H,W=L.shape[:2]
    X=_cuts(_lines(L,1),cols,W);Y=_cuts(_lines(L,0),rows,H)
    B=[_trim(L,(y,x)) for y in Y for x in X]
    # a table in front of the figure is cut away: its top edge must be flat below and either very flat or seen at the same height
    # in another panel of the same row (the sheets are drawn as a set)
    C=[_table_lines(L[y0:y1,x0:x1]) for (y0,y1),(x0,x1) in B];out=[]
    for i,((y0,y1),(x0,x1)) in enumerate(B):
        r=i//cols;others=[c for j in range(r*cols,r*cols+cols) if j!=i for c in C[j]];cut=None
        for y,flat in C[i]:
            if flat>=.8 or (flat>=.6 and any(abs(y-yo)<=3 for yo,_ in others)):cut=y;break
        out.append(((y0,y0+cut-3),(x0,x1)) if cut else ((y0,y1),(x0,x1)))
    return out

# ---------------------------------------------------------------- judges
_sess={}
def _model(name):
    if name not in _sess:
        import onnxruntime as ort
        os.makedirs(CACHE,exist_ok=True);p=os.path.join(CACHE,name+'.onnx')
        if not os.path.exists(p):
            print(f'downloading {name} (once)…',file=sys.stderr);urllib.request.urlretrieve(MODEL_URL.format(name),p+'.part');os.replace(p+'.part',p)
        so=ort.SessionOptions();so.enable_cpu_mem_arena=False
        _sess[name]=ort.InferenceSession(p,so,providers=['CPUExecutionProvider'])
    return _sess[name]
def neural(p,name):
    key=hashlib.sha1(p.tobytes()+name.encode()+str(p.shape).encode()).hexdigest()[:20];cp=os.path.join(CACHE,'maps',key+'.npy')
    if os.path.exists(cp):return np.load(cp).astype(np.float32)
    s=_model(name);sz=MODELS[name]
    x=cv2.resize(cv2.cvtColor(p,cv2.COLOR_BGR2RGB),(sz,sz),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    x=(x-[.485,.456,.406])/[.229,.224,.225] if name=='u2net' else x-.5
    o=s.run(None,{s.get_inputs()[0].name:x.transpose(2,0,1)[None].astype(np.float32)})[0][0,0]
    o=(o-o.min())/(o.max()-o.min()+1e-8);o=cv2.resize(o,(p.shape[1],p.shape[0]),interpolation=cv2.INTER_CUBIC).clip(0,1)
    os.makedirs(os.path.dirname(cp),exist_ok=True);np.save(cp,o.astype(np.float16));return o.astype(np.float32)
def backdrop(p,sure=None):
    """a smooth model of the painted backdrop and each pixel's distance to it, in noise units.
    Fitted where both neural judges are sure there is backdrop (else on the panel's rim), outliers refused."""
    h,w=p.shape[:2];L=lab(p);bm=np.zeros((h,w),bool);r=max(3,h//80);bm[:r]=True;bm[:,:r]=True;bm[:,-r:]=True
    if sure is not None and sure.sum()>h*w*.03:bm=sure.copy()
    if bm.sum()>40000:bm&=(np.random.default_rng(0).random((h,w))<40000/bm.sum())
    Y,X=np.mgrid[0:h,0:w];Xn=X/w;Yn=Y/h;AA=np.stack([np.ones_like(Xn),Xn,Yn,Xn*Yn,Xn**2,Yn**2,Yn**3],-1);A=AA[bm]
    ok=np.ones(len(A),bool);bg=np.zeros_like(L)
    for _ in range(5):
        coef=[np.linalg.lstsq(A[ok],L[...,c][bm][ok],rcond=None)[0] for c in range(3)]
        res=np.sqrt(sum((L[...,c][bm]-A@coef[c])**2*(.6 if c==0 else 1.6) for c in range(3)))
        sd=np.median(res[ok])*1.48+.5;ok=res<3*sd
    for c in range(3):bg[...,c]=AA@coef[c]
    d=np.sqrt(((L-bg)**2*np.array([.6,1.6,1.6])).sum(-1));d=cv2.GaussianBlur(d,(3,3),0)
    return bg,d/sd,ok.mean()
def judge_backdrop(p,sure=None,k=3.5):
    """1 where the panel is not backdrop reachable from an edge, 0 where it is"""
    bg,z,_=backdrop(p,sure);cand=(z<k).astype(np.uint8);h,w=cand.shape
    n,l=cv2.connectedComponents(cand,connectivity=4)
    edge=set(np.unique(np.r_[l[0],l[-1],l[:,0],l[:,-1]]))-{0}
    reach=np.isin(l,list(edge))
    return (~reach).astype(np.float32),z,bg

# ---------------------------------------------------------------- outline and alpha
K=np.ones((3,3),np.uint8)
def open2(m):
    """an exact 2x2 opening (OpenCV's own shifts the result by a pixel with an even kernel)"""
    k2=np.ones((2,2),np.uint8);return cv2.dilate(cv2.erode(m,k2,anchor=(0,0)),k2,anchor=(1,1))
def refine_outline(m,J,k=3.5,r=4):
    """the neural judges see shapes, not pixels: within a few pixels of the outline the backdrop model decides.
    Backdrop-like pixels joined to the backdrop leave; ink and figure pixels just outside come back in."""
    z=J['z'];band=(cv2.dilate(m,K,iterations=r)>0)&(cv2.erode(m,K,iterations=r)==0)
    sh=J['sh'];L=J['lab'];W=np.array([.6,1.6,1.6],np.float32)
    # within the band, a pixel nearer to the backdrop seen just outside than to the figure just inside belongs to the backdrop
    # (a painted drop shadow, a soft halo): two local colours, not one global model
    core=cv2.erode(m,K,iterations=r+1)>0;sure=(m==0)&~band&((z<k)|sh)
    Fl=_near_mean(L,core.astype(np.float32),(5,9,17,33));Bl=_near_mean(L,sure.astype(np.float32),(5,9,17,33,65))
    dF=np.sqrt((((L-Fl)**2)*W).sum(-1));dB=np.sqrt((((L-Bl)**2)*W).sum(-1))
    sureF=(J['u']>.9)&(J['s']>.9)  # both neural judges certain: the band may not cut there (a grey beak on a grey wall)
    bgl=((z<k)|sh|(band&(dB<dF)&(z<4*k)))&~(sureF&band)
    # backdrop-like pixels of the band, joined (through backdrop-like pixels) to the settled backdrop, leave
    seed=((m==0)&bgl&~band).astype(np.uint8);cand=(bgl&(band|(m==0))).astype(np.uint8)
    n,l=cv2.connectedComponents(cand,connectivity=4);ids=np.unique(l[seed>0]);ids=ids[ids>0]
    gone=np.isin(l,ids)&band;out=m.copy();out[gone]=0
    sm=(cv2.medianBlur(out*255,5)>127)&~gone;out[band]=sm[band]  # JPEG speckle: the band decides by majority, not pixel by pixel
    # figure-like pixels of the band, next to the figure and seen by a neural judge at all, come in (the ink line)
    grow=band&(z>2*k)&~sh&((J['u']>.15)|(J['s']>.15))&(out==0)  # clearly not backdrop: JPEG ringing stays out
    for _ in range(r):
        add=grow&(cv2.dilate(out,K)>0)
        if not add.any():break
        out[add]=1;grow&=~add
    out=open2(out)  # threads one pixel thin, detached from the body, go
    return keep_figure(out)
def _near_mean(I,m,sizes):
    out=None;have=None
    for kk in sizes:
        num=cv2.blur(I*m[...,None],(kk,kk));den=cv2.blur(m.astype(np.float32),(kk,kk))[...,None];e=num/np.maximum(den,1e-4)
        out=e if out is None else np.where(have,out,e);have=den>.02 if have is None else have|(den>.02)
    return out
def alpha_matte(p,m,J):
    """on the outline (one pixel each side), each pixel is read as a mix of the figure just inside (its ink and paint) and of what
    lies just outside (backdrop, shadow or table, as it really is there): this gives a true antialiased edge, and the colour kept
    is the figure's own, never the backdrop's"""
    I=p.astype(np.float32);inner=m.astype(bool);core=cv2.erode(m,K)>0;outer=~(cv2.dilate(m,K)>0)
    ring=(inner&~core)|((cv2.dilate(m,K)>0)&~inner)
    F=_near_mean(I,core.astype(np.float32),(3,5,9,17));B=_near_mean(I,outer.astype(np.float32),(3,5,9,17,33))
    D=F-B;a=((I-B)*D).sum(-1)/np.maximum((D*D).sum(-1),1.0);res=np.sqrt((((I-B)-a[...,None]*D)**2).sum(-1))
    a=np.clip(a,0,1);ok=(np.sqrt((D*D).sum(-1))>20)&(res<35)
    A=inner.astype(np.float32);A[ring&ok]=a[ring&ok]
    dec=np.clip((I-(1-a[...,None])*B)/np.maximum(a[...,None],.05),0,255)
    sel=ring&ok;C=I.copy();C[sel]=np.where((a[sel]>.6)[:,None],dec[sel],F[sel])  # an ink pixel the mix cannot explain keeps its own colour
    return A,np.clip(C,0,255).astype(np.uint8)

# ---------------------------------------------------------------- one panel
def keep_figure(m,min_frac=.02):
    """drop specks; keep every sizeable part (two figures can share a panel)"""
    n,l,st,_=cv2.connectedComponentsWithStats(m.astype(np.uint8),connectivity=8)
    if n<=1:return m.astype(np.uint8)
    big=st[1:,4].max();keep=[i for i in range(1,n) if st[i,4]>=max(big*.08,m.size*min_frac*.1)]
    return np.isin(l,keep).astype(np.uint8)
def fill_holes(m,z,k):
    """holes inside the figure are filled, unless they look like the backdrop (a pocket under an arm)"""
    inv=(1-m).astype(np.uint8);n,l,st,_=cv2.connectedComponentsWithStats(inv,connectivity=4);h,w=m.shape;out=m.copy()
    for i in range(1,n):
        x,y,ww,hh,ar=st[i]
        if x==0 or y==0 or x+ww==w or y+hh==h:continue
        sel=l==i
        if np.median(z[sel])>k or ar<30:out[sel]=1
    return out
def decide(J,k=3.5):
    """two against one, with one exception: the neural judges are trained on people and both tend to drop what a person wears or
    carries (wings, a cloud for a head, a receiver). What the backdrop judge sees as plainly not backdrop, attached to the figure,
    comes back — unless it has the shape of a table: a wide band lying along the bottom edge."""
    m=(J['v']>=2).astype(np.uint8);h,w=m.shape
    L0=J['lab'][...,0];lap=np.abs(cv2.Laplacian(cv2.GaussianBlur(L0,(0,0),.8),cv2.CV_32F))
    flat=max(.6,3*float(np.median(lap[(J['v']==0)])))  # how smooth the real backdrop is, in this panel
    cand=((J['c']>0)&(J['z']>2*k)&~J['sh']&(m==0)).astype(np.uint8);cand=cv2.morphologyEx(cand,cv2.MORPH_OPEN,K)
    n,l,st,_=cv2.connectedComponentsWithStats(cand,connectivity=4);near=cv2.dilate(m,K,iterations=2)>0
    for i in range(1,n):
        x,y,ww,hh,ar=st[i]
        if ar<h*w*.002:continue
        sel=l==i
        if not (sel&near).any():continue
        table=(y+hh>=h-max(3,h//60)) and ww>w*.3 and hh<h*.25
        drawn=np.median(lap[sel])>2*flat
        if drawn and not table:m[sel]=1
    # and the reverse: backdrop shut in by the figure (under an arm, between the legs) or its shadow, which every judge may keep.
    # It is backdrop-coloured (or the backdrop's shadow), sizeable and smooth: paint has texture, a shadowed wall has none.
    pk=(((J['z']<1.5)|J['sh'])&(m==1)&~((J['u']>.9)&(J['s']>.9))).astype(np.uint8);pk=cv2.morphologyEx(pk,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,l,st,_=cv2.connectedComponentsWithStats(pk,connectivity=4)
    for i in range(1,n):
        if st[i,4]<h*w*.003:continue
        sel=l==i;core=cv2.erode(sel.astype(np.uint8),np.ones((5,5),np.uint8))>0
        if core.sum()>=20 and np.median(lap[core])<flat:m[sel]=0
    # backdrop seen between legs or arms, open at the bottom of the panel, that the neural judges fill in: plain, backdrop-like,
    # reached from the bottom edge
    ob=((J['c']==0)&(J['z']<k)&(m==1)).astype(np.uint8);ob=cv2.morphologyEx(ob,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,l,st,_=cv2.connectedComponentsWithStats(ob,connectivity=4)
    for i in range(1,n):
        x,y,ww,hh,ar=st[i]
        if ar<h*w*.002 or y+hh<h-2:continue
        sel=l==i;core=cv2.erode(sel.astype(np.uint8),np.ones((3,3),np.uint8))>0
        if core.sum()>=10 and np.median(lap[core])<2*flat:m[sel]=0
    return m
def judge_panel(p,maps=None):
    u=neural(p,'u2net') if maps is None else maps[0];s=neural(p,'isnet-general-use') if maps is None else maps[1]
    sure=cv2.erode(((u<.1)&(s<.1)).astype(np.uint8),np.ones((5,5),np.uint8))>0
    c,z,bg=judge_backdrop(p,sure)
    votes=(u>.5).astype(np.float32)+(s>.5)+c
    L=lab(p)
    # a shadow cast on the backdrop: the backdrop's hue, darker. Only broad patches count (an ink line is thin and stays)
    sh=((np.abs(L[...,1]-bg[...,1])<4)&(np.abs(L[...,2]-bg[...,2])<5)&(L[...,0]<bg[...,0]+4)&(L[...,0]>bg[...,0]-40)).astype(np.uint8)
    sh=cv2.morphologyEx(sh,cv2.MORPH_OPEN,np.ones((4,4),np.uint8))>0
    return dict(v=votes,u=u,s=s,c=c,z=z,bg=bg,lab=L,sh=sh)

# ---------------------------------------------------------------- drawing a frame
def render(C,m,Mx,s,FW,FH,glow=None):
    """the outline is drawn at the final size, not enlarged: the mask becomes a signed distance field, smoothed by less than a pixel
    so the staircase of the source pixels melts into a curve, enlarged, then cut with one pixel of antialiasing at the final size.
    Colours just outside the figure are carried out from the figure, so the soft edge never shows the backdrop.
    glow: optional (b,g,r,strength,radius) light halo behind the figure."""
    m=(m>0).astype(np.uint8)
    din=cv2.distanceTransform(m,cv2.DIST_L2,5);dout=cv2.distanceTransform(1-m,cv2.DIST_L2,5)
    sdf=np.where(m>0,din-.5,-(dout-.5)).astype(np.float32);sdf=cv2.GaussianBlur(sdf,(0,0),.7)
    band=((1-m)>0)&(dout<=4);col=C.copy()
    if band.any():col=cv2.inpaint(C,band.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    big=cv2.warpAffine(sdf*s,Mx,(FW,FH),flags=cv2.INTER_CUBIC,borderValue=-99)
    A=np.clip(big/1.1+.5,0,1)
    rgb=cv2.warpAffine(col,Mx,(FW,FH),flags=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    if glow:
        b,g,r,k,rad=glow;G=np.clip(cv2.GaussianBlur(A,(0,0),rad)*k,0,1)*(1-A);outA=A+G
        rgb=(rgb*A[...,None]+np.array([b,g,r],np.float32)*G[...,None])/np.maximum(outA,1e-4)[...,None];A=outA
    f=np.dstack([np.clip(rgb,0,255),A*255]).astype(np.uint8);f[...,:3][f[...,3]==0]=0
    return f

# ---------------------------------------------------------------- the sheet
def headbox(m):
    ys,xs=np.where(m>.5);top=ys.min();H=m.shape[0];band=m[top:top+int(H*.16)]>.5;cols=np.where(band.any(0))[0]
    return top,cols.min(),cols.max()
def match_scale(m0,h0,m,h):
    """how much panel m must be scaled to match panel m0, on the head and shoulders"""
    t0,l0,r0=h0;t,l,r=h;H0,W0=m0.shape;band=int((H0-t0)*.45);ref=m0[t0:t0+band].astype(np.float32);cx0=(l0+r0)/2;cx=(l+r)/2
    best=(-1,1.0)
    for k in np.arange(.6,1.62,.02):
        for dx in range(-10,11,2):
            A=np.float32([[k,0,cx0+dx-cx*k],[0,k,-t*k]]);w=cv2.warpAffine(m.astype(np.float32),A,(W0,band),flags=cv2.INTER_LINEAR)
            inter=np.minimum(w,ref).sum();uni=np.maximum(w,ref).sum()+1e-6;iou=inter/uni
            if iou>best[0]:best=(iou,float(k))
    return best[1]
def shoulders(m,t):
    y=int(t+(m.shape[0]-t)*.42);xs=np.where(m[y]>.5)[0];return (xs.max()-xs.min()) if len(xs) else 1
def run(src,out,rows=2,cols=4,FH=472,pad=10,proof=None,report=None,maps=None,verbose=True,glow=None):
    im=cv2.imread(src,cv2.IMREAD_COLOR)
    if im is None:raise SystemExit(f'cannot read {src}')
    boxes=find_panels(im,rows,cols);P=[im[y0:y1,x0:x1].copy() for (y0,y1),(x0,x1) in boxes];N=len(P)
    J=[judge_panel(p,None if maps is None else maps[i]) for i,p in enumerate(P)]
    M=[decide(j) for j in J]
    # the other panels, aligned on this one, settle what the three judges left uncertain (2 against 1)
    hb=[headbox(m) for m in M];ref=[]
    for i in range(N):
        t,l,r=hb[i];ref.append(((l+r)/2,t,max(r-l,1)))
    doubts=[]
    for i in range(N):
        v,z=J[i]['v'],J[i]['z'];h,w=v.shape;cx,t,hw=ref[i];acc=np.zeros((h,w),np.float32);cnt=0
        for j in range(N):
            if j==i:continue
            cxj,tj,hwj=ref[j];k=hw/hwj;A=np.float32([[k,0,cx-cxj*k],[0,k,t-tj*k]])
            acc+=cv2.warpAffine(M[j].astype(np.float32),A,(w,h),flags=cv2.INTER_LINEAR,borderValue=0);cnt+=1
        cons=acc/cnt
        split=(v==1)|(v==2)  # one judge against two
        m=M[i].copy();flip=np.zeros_like(split)  # the others are witnesses, never judges: an effort frame may differ on purpose
        m=keep_figure(m);m=fill_holes(m,z,3.5);m=open2(m)
        M[i]=m
        # what remains doubtful: three-way splits the others could not settle, and lone disagreements with the others
        lone=((m==1)&(cons<.1))|((m==0)&(cons>.9));lone=cv2.morphologyEx(lone.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        d1=int(split.sum()-flip.sum());doubts.append({'panel':i,'split_px':d1,'lonely_px':int(lone.sum()),'lonely':lone})
    # alpha and colour
    RGBA=[];
    for i,p in enumerate(P):
        M[i]=refine_outline(M[i],J[i]);A,C=alpha_matte(p,M[i],J[i]);M[i]=(M[i]>0)&(A>=.5);M[i]=keep_figure(M[i].astype(np.uint8));RGBA.append(np.dstack([C,(A*255).astype(np.uint8)]))  # the shape follows the matte: a rim pixel that is mostly backdrop is outside
    # one scale for all, corrected per row on head and shoulders, the top of the head on one line
    hb=[headbox(r[...,3]/255) for r in RGBA];sw=[shoulders(r[...,3]/255,hb[i][0]) for i,r in enumerate(RGBA)]
    t0=hb[0][0];ph=P[0].shape[0];s0=(FH-14)/(ph-t0);T=[]
    # the size of each panel relative to the first, found by matching the silhouette of the head (its top part, down to the
    # shoulders) over a range of scales and small shifts; one value per row (the median), since a sheet's rows are drawn as sets
    ks=[match_scale(M[0],hb[0],M[i],hb[i]) for i in range(N)]
    for i in range(N):
        rr=i//cols;k=float(np.median(ks[rr*cols:rr*cols+cols]));t,l,r=hb[i];T.append((s0*k,(l+r)/2,t))
    half=max(max((cx-np.where(R[...,3]>8)[1].min())*s,(np.where(R[...,3]>8)[1].max()-cx)*s) for R,(s,cx,t) in zip(RGBA,T))
    FW=max(200,int(np.ceil((half+pad)*2/4)*4));sheet=np.zeros((FH,FW*N,4),np.uint8)
    for i,(R,(s,cx,t)) in enumerate(zip(RGBA,T)):
        Mx=np.float32([[s,0,FW/2-cx*s],[0,s,1-t*s]])
        sheet[:,i*FW:(i+1)*FW]=render(R[...,:3],M[i],Mx,s,FW,FH,glow)
    ext=os.path.splitext(out)[1].lower();cv2.imwrite(out,sheet,[cv2.IMWRITE_WEBP_QUALITY,92] if ext=='.webp' else [])
    if proof:make_proof(sheet,N,FW,proof,doubts,T,boxes)
    rep={'source':src,'frames':N,'frame_width':FW,'frame_height':FH,'panels':[[list(map(int,b[0])),list(map(int,b[1]))] for b in boxes],
         'doubts':[{k:v for k,v in d.items() if k!='lonely'} for d in doubts],
         'transforms':[[round(float(a),4) for a in t] for t in T]}
    if report:json.dump(rep,open(report,'w'),indent=1)
    if verbose:print(json.dumps({k:rep[k] for k in ('frames','frame_width')}),'doubts:',[(d['panel'],d['split_px'],d['lonely_px']) for d in doubts])
    return rep,sheet,M,P,J
def make_proof(sheet,N,FW,out,doubts,T,boxes):
    """every frame at full size, on magenta and on white, with what the program doubts outlined in yellow"""
    cols=[]
    for i in range(N):
        f=sheet[:,i*FW:(i+1)*FW].astype(np.float32);a=f[...,3:]/255
        mg=f[...,:3]*a+np.array([200,0,200])*(1-a);wh=f[...,:3]*a+255*(1-a)
        col=np.vstack([mg,wh]).astype(np.uint8)
        lone=doubts[i]['lonely']
        if lone.any():
            s,cx,t=T[i];Mx=np.float32([[s,0,FW/2-cx*s],[0,s,1-t*s]]);lw=cv2.warpAffine(lone*255,Mx,(FW,sheet.shape[0]))
            cs,_=cv2.findContours((lw>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for k in (0,sheet.shape[0]):cv2.drawContours(col,[c+np.array([0,k]) for c in cs],-1,(0,255,255),1)
        cols.append(col)
    cv2.imwrite(out,np.hstack(cols))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sheet');ap.add_argument('-o','--out',required=True)
    ap.add_argument('--grid',default='4x2',help='columns x rows (default 4x2)');ap.add_argument('--height',type=int,default=472)
    ap.add_argument('--proof');ap.add_argument('--report')
    ap.add_argument('--glow',help='light halo behind the figure: B,G,R,strength,radius (e.g. 255,245,230,.5,6)')
    a=ap.parse_args();c,r=map(int,a.grid.lower().split('x'))
    run(a.sheet,a.out,rows=r,cols=c,FH=a.height,proof=a.proof,report=a.report,glow=tuple(map(float,a.glow.split(','))) if a.glow else None)
