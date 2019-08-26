import cv2
import numpy as np
import time
import argparse
import json
import os
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
ref_pt = []
cropping = False
c=False
r=.1

def ims(name,im):                     
     oim=np.float32(im)-np.min(im)
     oim/=np.max(oim)              
     cv2.imshow(name,np.uint8(255*oim))
     cv2.waitKey(22) 

def ate(im):
    oim=np.float32(im)-np.min(im)
    oim/=np.max(oim)              
    return np.uint8(255*oim)

def grate(im):
    ma=np.max(np.abs(im))
    oim=np.float32(im)
    oim/=ma           
    return np.uint8(127*oim+127)
    
def binarize(im):
    oim=im.copy()
    msk=oim>np.mean(oim)
    mu1=np.mean(oim[msk])
    mu2=np.mean(oim[msk==False])
    sig1=np.std(oim[msk])
    sig2=np.std(oim[msk])
    map1=(1/sig1)*np.exp(-((oim-mu1)**2/sig1**2))
    map2=(1/sig2)*np.exp(-((oim-mu2)**2/sig2**2))
    new_msk=map1>map2
    changed=np.sum(np.sum(new_msk==msk))!=np.product(msk.shape)
    while changed:
        msk=new_msk.copy()
        mu1=np.mean(oim[msk])
        mu2=np.mean(oim[msk==False])
        sig1=np.std(oim[msk])
        sig2=np.std(oim[msk])
        map1=(1/sig1)*np.exp(-((oim-mu1)**2/sig1**2))
        map2=(1/sig2)*np.exp(-((oim-mu2)**2/sig2**2))
        new_msk=map1>map2
        changed=np.sum(np.sum(new_msk==msk))!=np.product(msk.shape)
    return new_msk

    
def digitize(segment):                
    bits=[]
    #segment=binarize(segment)
    for i in range(7):
        x_ind=[1,0,2,1,0,2,1][i]
        y_ind=[0,1,1,2,3,3,4][i]
        x_off=x_ind*10
        y_off=[0,10,55,65,110][y_ind]
        width=10
        height=[10,45,10,45,10][y_ind]
        m=np.mean(segment[y_off:y_off+height,x_off:x_off+width])
        bits.append(m<-4)
    return bits

def render(digit):                    
    oim=np.zeros((120,30))
    for i,bit in enumerate(digit):
        if bit:                 
            x_ind=[1,0,2,1,0,2,1][i]
            y_ind=[0,1,1,2,3,3,4][i]
            x_off=x_ind*10           
            y_off=[0,10,55,65,110][y_ind]
            width=10                  
            height=[10,45,10,45,10][y_ind]                      
            oim[y_off:y_off+height,x_off:x_off+width]=1
    #ims(name,oim)
    return oim

def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
        
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
            
    return dst

def get_clock(image,points):
    t1=np.array(points)
    t2=np.array([[0,0],[230,0],[230,120],[0,120]])
    r1=cv2.boundingRect(np.float32([t1]))          
    r2=cv2.boundingRect(np.float32([t2]))
    img1Rect = image[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])                 
    t1Rect = []                         
    t2Rect = []                         
    for j in range(3):                   
        t1Rect.append(((t1[j][0] - r1[0]),(t1[j][1] - r1[1])))
        t2Rect.append(((t2[j][0] - r2[0]),(t2[j][1] - r2[1])))
    img1Rect = image[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]] 
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    g=cv2.cvtColor(warpImage1,cv2.COLOR_BGR2GRAY)
    #gg1=cv2.GaussianBlur(g,(9,9),2)
    #gg3=cv2.GaussianBlur(g,(23,23),5)
    gg1=cv2.GaussianBlur(g,(7,7),1)
    gg3=cv2.GaussianBlur(g,(45,45),13)
    #return binarize(np.float32(gg1)-gg3)
    #return binarize(gg1)
    return (np.float32(gg1)-gg3)

def get_digits(clock,patterns):
    num=0
    oim=clock.copy()
    start=True
    #dim=clock
    for i in range(2,6):
        dig=oim[:,i*40:i*40+30]
        bits=np.array(digitize(dig))
        #render('bbit'+str(i),bits)
        #ims(str(i),dig>np.mean(dig))
        ind=np.where(np.sum(bits.reshape(1,7)==patterns,1)==7)[0]
        if not len(ind) and np.sum(bits):
            #print("oops")
            return -1
        elif not np.sum(bits):
            num=num*10
        elif len(ind):
            start=True
            #print(str(i)+' '+str(ind[0]))
            num=num*10+ind[0]
    #if has_dot(oim):
    #    num/=10
    return num

def has_dot(oim):
    return np.mean(oim[110:120,195:205])<0

def update_ref(pts,tag):
    opts=np.float32(pts)
    inds=[]
    vec=np.ones(2)
    if tag==ord('g') or tag==ord('b'):
        dphi=np.pi/180
        if tag==ord('b'):
            dphi*=-1
        rotmat=np.array([[np.cos(dphi),-np.sin(dphi)],[np.sin(dphi), np.cos(dphi)]])
        c=np.mean(opts,0).reshape(1,2)
        opts=np.dot(opts-c,rotmat)+c
    else:
        if tag==ord('a') or tag==ord('z') or tag== ord('d') or tag==ord('c'):
            vec[:]=opts[0,:]-opts[3,:]+opts[1,:]-opts[2,:]
            vec/=np.sqrt(np.sum(np.square(vec)))
            if tag==ord('a') or tag==ord('z'):
                inds=[0,1]
            else:
                inds=[2,3]
            if tag==ord('z') or tag==ord('d'):
                vec*=-1
        elif tag == ord('s') or tag==ord('x') or tag == ord('f') or tag==ord('v'):
            vec[:]=opts[1,:]-opts[0,:]+opts[2,:]-opts[3,:]
            vec/=np.sqrt(np.sum(np.square(vec)))
            if tag==ord('s') or tag==ord('x'):
                inds=[1,2]
            else:
                inds=[0,3]
            if tag==ord('x') or tag==ord('f'):
                vec*=-1
        opts[inds,:]+=vec
    return [[x for x in y] for y in opts]
    
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global ref_pt, cropping
    
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_pt.append( [x, y])
        #cropping = True
        print ("down"+str([x,y]))
        print(len(ref_pt))
    # check to see if the left mouse button was released
    #elif event == cv2.EVENT_LBUTTONUP:
    #    # record the ending (x, y) coordinates and indicate that
    #    # the cropping operation is finished
    #    refPt.append((x, y))
    #    cropping = False
    #print('i')
    #if len(refPt)==4:
    #    c=True
    #    #print ("UP")
    #    # draw a rectangle around the region of interest
    #    cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
    #    cv2.rectangle(image, refPt[1], refPt[2], (0, 255, 0), 2)
    #    cv2.rectangle(image, refPt[2], refPt[3], (0, 255, 0), 2)
    #    cv2.rectangle(image, refPt[3], refPt[0], (0, 255, 0), 2)
    #    cv2.imshow("image", image)



def get_ref_pts(ref_pt):
    while len(ref_pt)<4:
        #print(len(ref_pt))
        # display the image and wait for a keypress
        ret,image=cap.read()
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("r"):
            print ('redo')
            
            ref_pt=[]
            
        
	    # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, help="Path to the input video")
ap.add_argument("-t", "--top_name", required=True, help="name of the top")
ap.add_argument("-r", "--reuse_rectangle", action='store_true', help="Reuse previous rectangle")
args = vars(ap.parse_args())

meta_data=json.load(open('tops/data.json'))

# load the image, clone it, and setup the mouse callback function
#image = cv2.imread(args["image"])
#clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
cap=cv2.VideoCapture(0)
ret,image=cap.read()
# keep looping until the 'q' key is pressed

if args["reuse_rectangle"]:
    ref_pt=meta_data['prev_rect']
else:
    ref_pt=[]
    get_ref_pts(ref_pt)
print('got')
meta_data['prev_rect']=ref_pt
f=open('tops/data.json','w')
f.write(json.dumps(meta_data))
f.close()
meta_data=json.load(open('tops/data.json'))


clock=get_clock(image,ref_pt)
dot=clock[-10:,-40:-30]
mu=np.mean(clock)
patterns=np.array([[1,1,1,0,1,1,1],
                   [0,0,1,0,0,1,0],
                   [1,0,1,1,1,0,1],
                   [1,0,1,1,0,1,1],
                   [0,1,1,1,0,1,0],
                   [1,1,0,1,0,1,1],
                   [1,1,0,1,1,1,1],
                   [1,1,1,0,0,1,0],
                   [1,1,1,1,1,1,1],
                   [1,1,1,1,0,1,1]])
rpms=[]
timestamps=[]
print ('press t to start, r to redo, a/z for ud top, s/x for rl right, d/c for du bottom f/v for lr left g/b for clockwise,counterclockwise')
while True:
    ret,image=cap.read()
    
    clock=get_clock(image,ref_pt)
    for i in range(6):
        digit=digitize(clock[:,i*40:i*40+30])
        #print(digit)
        image[121:241,i*40:i*40+30,:]=render(digit).reshape(120,30,1)*255
    dot=dot*(1-r)+clock[-10:,-40:-30]*r
    digits=get_digits(clock,patterns)
    if np.mean(dot)<-.1:
        digits/=10
        image[231:241,190:200,:]=255
    #ims('clock',clock)
    key = cv2.waitKey(1) & 0xFF
    image[:121,:231,:]=grate(clock.reshape(121,231,1))
    #print(digits)
    #image[:121,:231,:]=255*clock.reshape(121,231,1)
    image[[10,55,65,110],:231,:]=[[[0,255,0]]]
    image[:121,np.arange(24)*10,:]=[[[0,255,0]]]
    #for i in range(6):
    #    digit=digitize(clock[:,i*40:i*40+30])
    #    print(digit)
    #    image[121:241,i*40:i*40+30,:]=render(digit).reshape(120,30,1)*255
    image=cv2.putText(image,str(digits),(400,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),4,cv2.LINE_AA)
    if key == ord("t"):
        print('end')
        break
    elif key == ord("r"):
        print('redo rectangle')
        ref_pt=[]
        get_ref_pts(ref_pt)
    elif key == ord('a') or key == ord('z') or key == ord('s') or key==ord('x') or key==ord('d') or key==ord('c') or key==ord('f') or key==ord('v') or key==ord('g') or key==ord('b'):
        ref_pt=update_ref(ref_pt,key)
    cv2.imshow("image", image)
        
        
start_time=time.time()
num_0=0
while True:
    ret,image=cap.read()
    t=time.time()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a') or key == ord('z') or key == ord('s') or key==ord('x') or key==ord('d') or key==ord('c') or key==ord('f') or key==ord('v') or key==ord('g') or key==ord('b'):
        ref_pt=update_ref(ref_pt,key)
    elif key == ord("r"):
        print('redo rectangle')
        get_ref_pts(ref_pt)
    clock=get_clock(image,ref_pt)
    for i in range(6):
        digit=digitize(clock[:,i*40:i*40+30])
        image[121:241,i*40:i*40+30,:]=render(digit).reshape(120,30,1)*255
    dot=dot*(1-r)+clock[-10:,-40:-30]*r
    
    #mu=mu*(1-r)+np.mean(clock)*r
    digits=get_digits(clock,patterns)
    if np.mean(dot)<0:
        image[231:241,190:200,:]=255
    #    digits/=10
    #ims('clock',clock)
    image[:121,:231,:]=grate(clock.reshape(121,231,1))
    image[[10,55,65,110],:231,:]=[[[0,255,0]]]
    image[:121,np.arange(24)*10,:]=[[[0,255,0]]]
    #for i in range(6):
    #    digit=digitize(clock[:,i*40:i*40+30])
    #    image[121:241,i*40:i*40+30,:]=render(digit).reshape(120,30,1)*255
    image=cv2.putText(image,str(digits),(400,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),4,cv2.LINE_AA)
    tt=np.int32(t-start_time)
    stopwatch=str(tt//60)+':'+str(tt%60)
    #+":"+str(len(rpms))
    image=cv2.putText(image,stopwatch,(400,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),4,cv2.LINE_AA)
    #for i,digit in enumerate(digits):
    #   image[121:241,i*40:i*40+30,:]=render(patterns[int(digit)]).reshape(120,30,1)
    #ims('mu',clock>)
    #print (digits)
    
    if len(rpms)>100:
        if rpms[-100]==digits:
             if np.sum(rpms[-100:]==digits)==100:
                  print(rpms[-100:])
                  print(digits)
                  break
    if digits>0:
        rpms.append(digits)
        timestamps.append(t-start_time)
    #img=cv2.putText(image,'OpenCV',(100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),4,cv2.LINE_AA)
    cv2.imshow("image", image)

top_name=args["top_name"]
names=meta_data['names']
out_dir='tops/'+top_name
if not os.path.exists(out_dir):
    os.system('mkdir '+out_dir)
if top_name in names.keys():
    np.save(out_dir+'/'+str(names[top_name])+'.npy',np.array([timestamps,rpms]))
else:
    np.save(out_dir+'/0.npy',np.array([timestamps,rpms]))

meta_data['prev_rect']=[[int(y) for y in x] for x in ref_pt]
if top_name in names.keys():
    names[top_name]=names[top_name]+1
else:
    names[top_name]=1
meta_data['names']=names
f=open('tops/data.json','w')
f.write(json.dumps(meta_data))
f.close()
