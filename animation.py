import pygame
import sys
import os
import imageio
import numpy as np
import math
import random

PI = math.pi


# ──────────────────────────────────────────────────────────────────────────────
#  PARTICLE SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

class Particle:
    __slots__ = ['x','y','vx','vy','color','life','max_life','size','gravity','ptype']

    def __init__(self, x, y, vx, vy, color, life, size, gravity=0.0, ptype='dot'):
        self.x, self.y     = float(x), float(y)
        self.vx, self.vy   = float(vx), float(vy)
        self.color         = color
        self.life          = life
        self.max_life      = life
        self.size          = float(size)
        self.gravity       = gravity
        self.ptype         = ptype

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += self.gravity
        self.vx *= 0.92
        self.life -= 1


class ParticleSystem:
    MAX = 600

    def __init__(self, w, h):
        self.particles = []
        self._surf = pygame.Surface((w, h), pygame.SRCALPHA)
        self.w, self.h = w, h

    def _emit(self, p):
        if len(self.particles) < self.MAX:
            self.particles.append(p)

    # ── emitters ──────────────────────────────────────────────────────────────

    def sparks(self, x, y, n=20, weapon=False):
        palette = ([(255,235,120),(255,210,60),(255,180,20),(255,255,190)]
                   if weapon else [(255,140,50),(255,100,30),(255,200,80),(220,90,20)])
        for _ in range(n):
            angle = random.uniform(0, 2*PI)
            spd   = random.uniform(3, 11)
            self._emit(Particle(x, y, math.cos(angle)*spd, math.sin(angle)*spd,
                                random.choice(palette),
                                random.randint(8,20), random.uniform(1.5,4),
                                gravity=0.28, ptype='spark'))

    def blood(self, x, y, n=14):
        for _ in range(n):
            angle = random.uniform(-PI, 0)
            spd   = random.uniform(2, 8)
            c     = random.choice([(185,20,20),(150,12,12),(160,15,15)])
            self._emit(Particle(x, y, math.cos(angle)*spd, math.sin(angle)*spd - 1.5,
                                c, random.randint(18,38), random.uniform(2,5.5),
                                gravity=0.38, ptype='blood'))

    def dust(self, x, y, direction=1, n=8):
        for _ in range(n):
            angle = random.uniform(PI*0.25, PI*0.75) * direction
            spd   = random.uniform(0.4, 2.5)
            g     = random.randint(115, 155)
            self._emit(Particle(x, y, math.cos(angle)*spd, math.sin(angle)*spd - 0.4,
                                (g, g-12, g-20), random.randint(22,44), random.uniform(3,9),
                                gravity=-0.04, ptype='dust'))

    def shockwave(self, x, y):
        for i in range(28):
            angle = (2*PI/28)*i
            spd   = random.uniform(5, 8)
            self._emit(Particle(x, y, math.cos(angle)*spd, math.sin(angle)*spd,
                                (255,255,220), random.randint(5,11), random.uniform(2,4),
                                gravity=0, ptype='spark'))

    def stars(self, x, y):
        for i in range(8):
            angle = (2*PI/8)*i
            spd   = random.uniform(5, 13)
            self._emit(Particle(x, y, math.cos(angle)*spd, math.sin(angle)*spd,
                                (255,255,180), random.randint(4,9), random.uniform(3,6),
                                gravity=0, ptype='star'))

    def sweat(self, x, y):
        for _ in range(2):
            self._emit(Particle(x + random.randint(-22,22), y + random.randint(-18,4),
                                random.uniform(-0.4,0.4), random.uniform(-2,-0.5),
                                (150,210,255), random.randint(14,26), random.uniform(2,4),
                                gravity=0.14, ptype='sweat'))

    def smoke(self, x, y, n=3):
        for _ in range(n):
            angle = random.uniform(-PI*2/3, -PI/3)
            spd   = random.uniform(0.3, 1.2)
            g     = random.randint(90, 130)
            self._emit(Particle(x + random.randint(-6,6), y,
                                math.cos(angle)*spd, math.sin(angle)*spd,
                                (g,g,g), random.randint(28,50), random.uniform(5,13),
                                gravity=-0.025, ptype='smoke'))

    # ── update / draw ─────────────────────────────────────────────────────────

    def update(self):
        self.particles = [p for p in self.particles if p.life > 0]
        for p in self.particles:
            p.update()

    def draw(self, surface):
        self._surf.fill((0,0,0,0))
        for p in self.particles:
            frac  = p.life / max(1, p.max_life)
            alpha = int(255 * frac)
            sz    = max(1, int(p.size * frac + 0.5))
            cx, cy = int(p.x), int(p.y)

            if p.ptype in ('dust', 'smoke', 'sweat'):
                a = alpha // 2
                pygame.draw.circle(self._surf, (*p.color, a), (cx,cy), sz)
            elif p.ptype == 'star':
                pygame.draw.circle(self._surf, (*p.color, alpha), (cx,cy), sz)
                arm = sz * 2
                pygame.draw.line(self._surf, (*p.color, alpha//2),
                                 (cx-arm, cy), (cx+arm, cy), max(1,sz//2))
                pygame.draw.line(self._surf, (*p.color, alpha//2),
                                 (cx, cy-arm), (cx, cy+arm), max(1,sz//2))
            else:
                pygame.draw.circle(self._surf, (*p.color, alpha), (cx,cy), sz)

        surface.blit(self._surf, (0,0))


# ──────────────────────────────────────────────────────────────────────────────
#  SCREEN SHAKE
# ──────────────────────────────────────────────────────────────────────────────

class ScreenShake:
    def __init__(self):
        self.amount = 0.0

    def trigger(self, strength):
        self.amount = max(self.amount, float(strength))

    def update(self):
        if self.amount > 0.4:
            off = (random.randint(-int(self.amount), int(self.amount)),
                   random.randint(-int(self.amount), int(self.amount)))
            self.amount *= 0.72
            return off
        self.amount = 0.0
        return (0, 0)


# ──────────────────────────────────────────────────────────────────────────────
#  SMOOTH HP TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class SmoothHP:
    def __init__(self, max_hp):
        self.display     = 1.0
        self.target      = 1.0
        self.max_hp      = max_hp
        self.flash_timer = 0

    def update(self, current_hp):
        self.target = max(0.0, current_hp / self.max_hp)
        diff = self.display - self.target
        if diff > 0.0005:
            self.display -= diff * 0.09 + 0.0018
            self.display  = max(self.target, self.display)
            if diff > 0.04:
                self.flash_timer = 9
        if self.flash_timer > 0:
            self.flash_timer -= 1


# ──────────────────────────────────────────────────────────────────────────────
#  COLOUR HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_c(c1, c2, t):
    return (int(lerp(c1[0],c2[0],t)),
            int(lerp(c1[1],c2[1],t)),
            int(lerp(c1[2],c2[2],t)))


# ──────────────────────────────────────────────────────────────────────────────
#  DRAWING PRIMITIVES
# ──────────────────────────────────────────────────────────────────────────────

def draw_limb(surf, main_c, shadow_c, start, end, w_start, w_end):
    """Tapered quad limb with shadow half + highlight strip."""
    sx, sy = start
    ex, ey = end
    angle  = math.atan2(ey-sy, ex-sx)
    perp   = angle + PI/2
    cp, sp = math.cos(perp), math.sin(perp)

    pts = [
        (sx + cp*w_start/2, sy + sp*w_start/2),
        (ex + cp*w_end/2,   ey + sp*w_end/2),
        (ex - cp*w_end/2,   ey - sp*w_end/2),
        (sx - cp*w_start/2, sy - sp*w_start/2),
    ]
    pygame.draw.polygon(surf, main_c, pts)

    # shadow side (half polygon)
    sh_pts = [
        start,
        end,
        (ex - cp*w_end/2,   ey - sp*w_end/2),
        (sx - cp*w_start/2, sy - sp*w_start/2),
    ]
    shadow_blend = lerp_c(main_c, shadow_c, 0.42)
    pygame.draw.polygon(surf, shadow_blend, sh_pts)

    # highlight strip
    hl = lerp_c(main_c, (255,255,255), 0.18)
    hl_pts = [
        (sx + cp*w_start*0.22, sy + sp*w_start*0.22),
        (ex + cp*w_end*0.22,   ey + sp*w_end*0.22),
        (ex + cp*w_end*0.44,   ey + sp*w_end*0.44),
        (sx + cp*w_start*0.44, sy + sp*w_start*0.44),
    ]
    pygame.draw.polygon(surf, hl, hl_pts)
    pygame.draw.polygon(surf, shadow_c, pts, 1)


def draw_fist(surf, x, y, skin_c, skin_s, attacking=False):
    x, y = int(x), int(y)
    if attacking:
        pts = [(x-11,y-7),(x+9,y-9),(x+13,y-2),(x+9,y+6),(x-11,y+5)]
    else:
        pts = [(x-8,y-5),(x+7,y-6),(x+10,y-1),(x+7,y+5),(x-8,y+4)]
    pygame.draw.polygon(surf, skin_c, pts)
    pygame.draw.polygon(surf, skin_s, pts, 1)
    # knuckle highlights
    hl = lerp_c(skin_c, (255,255,255), 0.28)
    for i in (-4, 0, 4):
        pygame.draw.circle(surf, hl, (x+2+i, y-3), 2)
    # thumb
    th = [(x-10,y-5),(x-14,y-10),(x-17,y-8),(x-13,y-3)]
    pygame.draw.polygon(surf, skin_c, th)
    pygame.draw.polygon(surf, skin_s, th, 1)


def draw_shoe(surf, pos, facing_right, shoe_c, sole_c=(30,30,30)):
    x, y = int(pos[0]), int(pos[1])
    f = 1 if facing_right else -1
    body = [(x-13,y-8),(x+14,y-8),(x+18+f*7,y-3),(x+18+f*7,y+2),(x-13,y+2)]
    pygame.draw.polygon(surf, shoe_c, body)
    sole = [(x-15,y+1),(x+21+f*7,y+1),(x+21+f*7,y+8),(x-15,y+8)]
    pygame.draw.polygon(surf, sole_c, sole)
    hl = lerp_c(shoe_c, (255,255,255), 0.28)
    pygame.draw.line(surf, hl, (x-8,y-7),(x+7,y-7), 2)
    for xi in range(-4,7,3):
        pygame.draw.circle(surf, (200,200,200), (x+xi, y-4), 1)


def draw_shadow_ellipse(surf, cx, foot_y, alpha_surf):
    alpha_surf.fill((0,0,0,0))
    pygame.draw.ellipse(alpha_surf, (0,0,0,55), (cx-32, int(foot_y)+2, 64, 13))
    surf.blit(alpha_surf, (0,0))


# ──────────────────────────────────────────────────────────────────────────────
#  CHARACTER PARTS
# ──────────────────────────────────────────────────────────────────────────────

SKIN    = (255,218,182)
SKIN_M  = (238,192,152)
SKIN_D  = (205,162,118)
SKIN_S  = (182,138,98)
SKIN_HL = (255,240,218)
LIPS    = (200,105,105)
EYE_W   = (245,245,245)
EYE_PUP = (14,14,14)
BLOOD_C = (185,20,20)
BLOOD_D = (120,10,10)


def draw_head(surf, cx, cy, facing_right, hair_c, eye_c,
              is_hit=False, mouth_open=False, angry=False, unconscious=False):
    cx, cy = int(cx), int(cy)
    f  = 1 if facing_right else -1
    hw, hh = 28, 34

    # neck
    neck_pts = [(cx-9,cy+8),(cx+9,cy+8),(cx+11,cy+30),(cx-11,cy+30)]
    pygame.draw.polygon(surf, SKIN_D, neck_pts)
    for xi in (-4, 4):
        pygame.draw.line(surf, lerp_c(SKIN_D,(0,0,0),0.18),
                         (cx+xi, cy+10),(cx+xi, cy+28), 1)

    # head base – draw as solid ellipse then overlay shadow half
    head_r = pygame.Rect(cx-hw, cy-hh, hw*2, hh*2)
    tmp = pygame.Surface((hw*2, hh*2), pygame.SRCALPHA)
    pygame.draw.ellipse(tmp, (*SKIN,255), (0,0,hw*2,hh*2))
    # shadow (right half)
    shaded = lerp_c(SKIN, SKIN_S, 0.32)
    pygame.draw.ellipse(tmp, (*shaded,160), (hw//2, 0, hw+hw//2, hh*2))
    surf.blit(tmp, (cx-hw, cy-hh))

    # jaw
    jaw_c   = lerp_c(SKIN, SKIN_M, 0.22)
    jaw_pts = [(cx-22,cy+8),(cx+22,cy+8),(cx+18,cy+26),(cx-18,cy+26)]
    pygame.draw.polygon(surf, jaw_c, jaw_pts)
    # subtle cheeks
    for side in (-1,1):
        pygame.draw.ellipse(surf, lerp_c(SKIN,(220,140,140),0.14),
                            (cx+side*12-7, cy+2, 14, 8))
    pygame.draw.ellipse(surf, SKIN_D, head_r, 2)

    # ears
    for side in (-1,1):
        ex2 = cx + side*(hw+1)
        ear = [(ex2-4,cy-6),(ex2+4*side,cy-6),(ex2+5*side,cy+6),(ex2-4,cy+6)]
        pygame.draw.polygon(surf, SKIN, ear)
        pygame.draw.ellipse(surf, lerp_c(SKIN,SKIN_S,0.38), (ex2-2+side,cy-3,5,8))

    # hair
    if not unconscious:
        hair_pts = [
            (cx-hw+3,cy-hh+14),(cx-hw+4,cy-hh+4),(cx-hw+8,cy-hh-2),
            (cx-6,cy-hh-8),(cx+2,cy-hh-11),(cx+10,cy-hh-9),
            (cx+hw-5,cy-hh-3),(cx+hw,cy-hh+8),(cx+hw+1,cy-hh+18),
            (cx+hw-3,cy-hh+14),(cx+hw-10,cy-4),(cx+hw-14,cy+2),
            (cx-hw+10,cy+0),(cx-hw+5,cy-4),
        ]
        pygame.draw.polygon(surf, hair_c, hair_pts)
        sheen = lerp_c(hair_c,(255,255,255),0.22)
        pygame.draw.ellipse(surf, sheen, (cx-10, cy-hh-6, 22, 12))
    else:
        h2 = [(cx-hw+3,cy-hh+14),(cx-hw+6,cy-hh-4),(cx-8,cy-hh-10),
              (cx+4,cy-hh-13),(cx+hw-4,cy-hh-5),(cx+hw+1,cy-hh+15),
              (cx+hw-8,cy-5),(cx,cy-8),(cx-hw+8,cy-3)]
        pygame.draw.polygon(surf, hair_c, h2)

    # eyes
    eye_y = cy - 7
    for side, ex2 in ((-1, cx-11),(1, cx+11)):
        socket_c = lerp_c(SKIN,SKIN_S,0.28)
        pygame.draw.ellipse(surf, socket_c, (ex2-9, eye_y-3, 18, 12))
        pygame.draw.ellipse(surf, EYE_W,    (ex2-7, eye_y-1, 14, 9))
        pygame.draw.ellipse(surf, (140,140,140), (ex2-7, eye_y-1, 14, 9), 1)
        if not unconscious:
            iris_x = ex2 + f*2
            pygame.draw.circle(surf, eye_c,  (int(iris_x), int(eye_y+4)), 4)
            pygame.draw.circle(surf, EYE_PUP,(int(iris_x), int(eye_y+4)), 2)
            pygame.draw.circle(surf, (255,255,255),(int(iris_x+1), int(eye_y+2)), 1)
            pygame.draw.arc(surf, lerp_c(SKIN,SKIN_S,0.18),
                            (ex2-7, eye_y-1, 14, 9), 0, PI, 1)
        else:
            pygame.draw.line(surf,(80,80,80),(ex2-5,eye_y),(ex2+5,eye_y+7),1)
            pygame.draw.line(surf,(80,80,80),(ex2+5,eye_y),(ex2-5,eye_y+7),1)
        pygame.draw.arc(surf,(28,28,28),(ex2-8,eye_y-2,16,10),PI/6,5*PI/6,2)

    # eyebrows
    brow_y = eye_y - 9
    for side, ex2 in ((-1,cx-11),(1,cx+11)):
        if angry or is_hit:
            inner = ex2 - side*9; outer = ex2 + side*9
            pygame.draw.line(surf, hair_c,
                             (int(inner), brow_y+side*2), (int(outer), brow_y-3), 3)
        else:
            pygame.draw.line(surf, hair_c, (ex2-8, brow_y), (ex2+8, brow_y-1), 2)
            pygame.draw.arc(surf, hair_c,  (ex2-9, brow_y-4, 18, 8), PI/12, 11*PI/12, 1)

    # nose
    nose_s = lerp_c(SKIN, SKIN_S, 0.3)
    nose_pts = [(cx+f*2,cy+1),(cx+f*4,cy+9),(cx+f*3,cy+12),
                (cx,cy+11),(cx-f*2,cy+12),(cx-f*1,cy+9)]
    pygame.draw.polygon(surf, nose_s, nose_pts)
    pygame.draw.circle(surf, lerp_c(SKIN,SKIN_S,0.5),(cx+f*3,cy+11), 3)

    # mouth
    mouth_y = cy + 19
    if mouth_open:
        pygame.draw.ellipse(surf,(42,14,14),(cx-8,mouth_y-3,16,11))
        pygame.draw.rect(surf,(232,232,232),(cx-6,mouth_y-2,12,4))
        for xi in (-2,2):
            pygame.draw.line(surf,(175,175,175),(cx+xi,mouth_y-2),(cx+xi,mouth_y+2),1)
        pygame.draw.arc(surf, LIPS,(cx-8,mouth_y-4,16,10),0,PI,2)
    elif is_hit:
        pts2 = [(cx-7,mouth_y+2),(cx-2,mouth_y),(cx+3,mouth_y+3),(cx+7,mouth_y-1)]
        pygame.draw.lines(surf,(155,60,60),False,pts2,2)
    else:
        pygame.draw.arc(surf,LIPS,(cx-7,mouth_y-2,14,7),PI/8,7*PI/8,2)
        pygame.draw.ellipse(surf,lerp_c(LIPS,SKIN,0.3),(cx-5,mouth_y+1,10,5))

    # hit flash
    if is_hit:
        tmp2 = pygame.Surface((hw*2,hh*2), pygame.SRCALPHA)
        pygame.draw.ellipse(tmp2,(255,80,80,75),(0,0,hw*2,hh*2))
        surf.blit(tmp2,(cx-hw,cy-hh))


def draw_torso(surf, cx, cy, facing_right, sc, sd, sl, is_hit=False):
    f  = 1 if facing_right else -1
    sw, sh = 36, 32
    pts = [(cx-sw,cy-sh),(cx+sw,cy-sh),(cx+sw*0.75,cy+sh),(cx-sw*0.75,cy+sh)]
    pygame.draw.polygon(surf, sc, pts)
    # shadow right side
    dark_pts = [(cx,cy-sh),(cx+sw,cy-sh),(cx+sw*0.75,cy+sh),(cx,cy+sh)]
    pygame.draw.polygon(surf, lerp_c(sc,sd,0.38), dark_pts)
    # center highlight
    pygame.draw.ellipse(surf, lerp_c(sc,sl,0.45), (cx-10,cy-sh+4,20,sh))
    pygame.draw.polygon(surf, sd, pts, 2)
    # v-neck
    collar = [(cx-13,cy-sh),(cx+13,cy-sh),(cx+f*6,cy-sh+22),(cx-f*4,cy-sh+22)]
    pygame.draw.polygon(surf, lerp_c(SKIN,SKIN_M,0.3), collar)
    pygame.draw.lines(surf, sd, False, collar[:3], 2)
    # chest lines
    for xi in (-8,8):
        pygame.draw.line(surf, lerp_c(sc,sd,0.22),(cx+xi,cy-sh+24),(cx+xi,cy+sh-8),1)
    # belt
    bt = cy+sh-10
    belt_pts = [(cx-sw*0.75,bt),(cx+sw*0.75,bt),(cx+sw*0.75,bt+11),(cx-sw*0.75,bt+11)]
    pygame.draw.polygon(surf,(55,42,32),belt_pts)
    pygame.draw.polygon(surf,(35,26,18),belt_pts,1)
    buckle = pygame.Rect(cx-7,bt+1,14,9)
    pygame.draw.rect(surf,(192,186,165),buckle,border_radius=2)
    pygame.draw.rect(surf,(110,105,90),buckle,1,border_radius=2)
    pygame.draw.circle(surf,lerp_c((192,186,165),(255,255,255),0.3),(cx,bt+5),3)
    if is_hit:
        tmp = pygame.Surface((sw*2+4,sh*2+4),pygame.SRCALPHA)
        pygame.draw.polygon(tmp,(255,200,100,75),
                            [(p[0]-(cx-sw-2),p[1]-(cy-sh-2)) for p in pts])
        surf.blit(tmp,(cx-sw-2,cy-sh-2))


def draw_pants(surf, pants_c, pants_d, l_hip, l_knee, l_foot, r_hip, r_knee, r_foot):
    for hip, knee, foot in ((l_hip,l_knee,l_foot),(r_hip,r_knee,r_foot)):
        pts = [(hip[0]-14,hip[1]),(hip[0]+14,hip[1]),
               (knee[0]+12,knee[1]),(foot[0]+13,foot[1]-6),
               (foot[0]-13,foot[1]-6),(knee[0]-12,knee[1])]
        pygame.draw.polygon(surf, pants_c, pts)
        sh_pts = [hip,(hip[0]+14,hip[1]),(knee[0]+12,knee[1]),
                  (foot[0]+13,foot[1]-6),(foot[0],foot[1]-6),knee]
        pygame.draw.polygon(surf, lerp_c(pants_c,pants_d,0.48), sh_pts)
        pygame.draw.line(surf, lerp_c(pants_c,pants_d,0.28),
                         (int(hip[0]-2),int(hip[1]+2)),(int(knee[0]-2),int(knee[1]-2)),1)
        khl = lerp_c(pants_c,(255,255,255),0.14)
        pygame.draw.circle(surf,khl,(int(knee[0]),int(knee[1])),5)
        pygame.draw.polygon(surf, pants_d, pts, 1)


def draw_arm(surf, shoulder, elbow, hand, skin_c, skin_s, sc, sd, sl,
             attacking=False):
    shoulder = tuple(int(v) for v in shoulder)
    elbow    = tuple(int(v) for v in elbow)
    hand     = tuple(int(v) for v in hand)
    # sleeve
    draw_limb(surf, sc, sd, shoulder, elbow, 20, 16)
    pygame.draw.line(surf, sd, (elbow[0]-8,elbow[1]),(elbow[0]+8,elbow[1]),2)
    # forearm
    draw_limb(surf, skin_c, skin_s, elbow, hand, 14, 11)
    pygame.draw.circle(surf, lerp_c(skin_c,skin_s,0.3), elbow, 7)
    pygame.draw.circle(surf, skin_s, elbow, 7, 1)
    # wrist detail
    wrist = ((elbow[0]+hand[0])//2,(elbow[1]+hand[1])//2)
    pygame.draw.line(surf, lerp_c(skin_c,(50,50,50),0.28),
                     (wrist[0]-5,wrist[1]),(wrist[0]+5,wrist[1]),2)
    draw_fist(surf, hand[0], hand[1], skin_c, skin_s, attacking)


def draw_weapon(surf, hand_pos, facing_right):
    """Detailed sword with glow."""
    f  = 1 if facing_right else -1
    hx, hy = int(hand_pos[0]), int(hand_pos[1])
    bs = (hx + f*8,  hy - 8)
    be = (hx + f*62, hy - 42)

    # blade glow
    glow = pygame.Surface((surf.get_width(), surf.get_height()), pygame.SRCALPHA)
    for gw in range(8,1,-2):
        pygame.draw.line(glow,(180,220,255,max(0,38-gw*4)),bs,be,gw*2)
    surf.blit(glow,(0,0))

    # blade body
    angle  = math.atan2(be[1]-bs[1], be[0]-bs[0])
    perp   = angle+PI/2
    cp,sp  = math.cos(perp), math.sin(perp)
    w1,w2  = 5,2
    blade_pts = [(bs[0]+cp*w1,bs[1]+sp*w1),(be[0]+cp*w2,be[1]+sp*w2),
                 (be[0]-cp*w2,be[1]-sp*w2),(bs[0]-cp*w1,bs[1]-sp*w1)]
    pygame.draw.polygon(surf,(185,190,200),blade_pts)
    pygame.draw.polygon(surf,(120,125,135),blade_pts,1)
    # edge hl
    pygame.draw.line(surf,(228,233,245),
                     (int(bs[0]+cp*w1*0.7),int(bs[1]+sp*w1*0.7)),
                     (int(be[0]+cp*w2*0.7),int(be[1]+sp*w2*0.7)),1)
    # tip
    tip = (hx+f*68, hy-46)
    pygame.draw.polygon(surf,(228,233,245),[be,(int(be[0]+cp*w2*1.5),int(be[1]+sp*w2*1.5)),tip])
    # fuller groove
    pygame.draw.line(surf,(120,125,135),(hx+f*15,hy-12),(hx+f*55,hy-39),1)
    # crossguard
    gd_pts = [(hx-15,hy-6),(hx+15,hy-6),(hx+12,hy+2),(hx-12,hy+2)]
    pygame.draw.polygon(surf,(120,125,135),gd_pts)
    pygame.draw.line(surf,(228,233,245),(hx-13,hy-4),(hx+13,hy-4),1)
    # grip
    grip_c = (68,44,24)
    pygame.draw.line(surf,grip_c,(hx-f*2,hy+2),(hx-f*2,hy+16),5)
    for yi in range(hy+2, hy+16, 3):
        wrap = lerp_c(grip_c,(118,84,48),0.5)
        pygame.draw.line(surf,wrap,(hx-f*2-3,yi),(hx-f*2+3,yi),1)
    # pommel
    pm = (hx-f*2, hy+19)
    pygame.draw.circle(surf,(120,125,135),pm,5)
    pygame.draw.circle(surf,(228,233,245),(pm[0]-1,pm[1]-1),2)
    pygame.draw.circle(surf,(90,95,105),pm,5,1)
    # shine
    mid = ((bs[0]+be[0])//2+f*5,(bs[1]+be[1])//2-3)
    gt = pygame.Surface((20,20),pygame.SRCALPHA)
    pygame.draw.circle(gt,(255,255,255,55),(10,10),8)
    surf.blit(gt,(mid[0]-10,mid[1]-10))


# ──────────────────────────────────────────────────────────────────────────────
#  POSE SYSTEM
# ──────────────────────────────────────────────────────────────────────────────

def idle_pose(cx, step):
    bob = math.sin(step * 0.18) * 2.5
    b   = bob
    return dict(
        cx=cx,
        head=(cx, 268+b*0.5),
        shoulder_l=(cx-32, 298+b*0.3), shoulder_r=(cx+32, 298+b*0.3),
        elbow_l=(cx-52, 315+b),        elbow_r=(cx+48, 310+b),
        hand_l=(cx-38, 282+b),         hand_r=(cx+58, 278+b),
        hip_l=(cx-16, 360),            hip_r=(cx+16, 360),
        knee_l=(cx-22, 398),           knee_r=(cx+18, 392),
        foot_l=(cx-32, 430),           foot_r=(cx+26, 430),
    )


def get_pose(action, is_agent, step, anim_t):
    """anim_t 0→1 is the action extension arc."""
    f  = 1 if is_agent else -1
    cx = 210 if is_agent else 690
    # mirror for opponent: negate all *f lateral offsets via final mirror step

    d = idle_pose(cx, step)
    if action is None or anim_t > 1.05:
        # Mirror laterally for opponent
        if not is_agent:
            d = _mirror(d, cx)
        return d

    e = math.sin(anim_t * PI)  # 0→1→0 extension arc

    if action == 0:   # PUNCH
        lean = e * 12
        d['head']       = (cx, 268 - e*6)
        d['shoulder_r'] = (cx+(32+e*18), 294-e*5)
        d['elbow_r']    = (cx+(48+e*30), 300-e*15)
        d['hand_r']     = (cx+(58+e*72), 272-e*18)
        d['hand_l']     = (cx-32,        275)
        d['foot_r']     = (cx+(26+lean*0.4), 430)
        d['knee_r']     = (cx+(18+lean*0.3), 392)

    elif action == 1:  # KICK
        d['head']   = (cx - e*4, 268+e*4)
        d['knee_r'] = (cx+(18+e*52), 378-e*42)
        d['foot_r'] = (cx+(26+e*92), 358-e*36)
        d['knee_l'] = (cx-(22+e*8),  402)
        d['elbow_l']= (cx-60, 310-e*10)
        d['hand_l'] = (cx-50, 278-e*5)
        d['elbow_r']= (cx+55, 305-e*5)
        d['hand_r'] = (cx+45, 273)

    elif action == 2:  # WEAPON
        d['head']       = (cx+e*4, 265-e*4)
        d['shoulder_r'] = (cx+(32+e*12), 292-e*8)
        d['elbow_r']    = (cx+(48+e*22), 290-e*20)
        d['hand_r']     = (cx+(58+e*56), 258-e*38)
        d['hand_l']     = (cx-28, 282)
        d['elbow_l']    = (cx-48, 308)

    if not is_agent:
        d = _mirror(d, cx)
    return d


def _mirror(d, cx):
    """Flip all x positions around cx for opponent."""
    def mx(x): return 2*cx - x
    def mp(p):  return (mx(p[0]), p[1])
    return {k: (mp(v) if isinstance(v,tuple) and len(v)==2 else v)
            for k,v in d.items()}


def get_death_pose(is_agent, death_t):
    cx  = 210 if is_agent else 690
    t   = min(1.0, death_t)
    dy  = t * 45
    dx  = t * 30 * (1 if is_agent else -1)
    return dict(
        cx=cx+dx,
        head=(cx+dx, 268+dy+t*32),
        shoulder_l=(cx+dx-32, 298+dy),
        shoulder_r=(cx+dx+32, 298+dy),
        elbow_l=(cx+dx-35,  320+dy+t*22),
        elbow_r=(cx+dx+35,  320+dy+t*22),
        hand_l=(cx+dx-55,   355+dy+t*32),
        hand_r=(cx+dx+55,   355+dy+t*32),
        hip_l=(cx+dx-16,    360+dy),
        hip_r=(cx+dx+16,    360+dy),
        knee_l=(cx+dx-22,   398+dy+t*22),
        knee_r=(cx+dx+18,   392+dy+t*22),
        foot_l=(cx+dx-32,   430+t*12),
        foot_r=(cx+dx+26,   430+t*12),
    )


# ──────────────────────────────────────────────────────────────────────────────
#  FULL CHARACTER DRAW
# ──────────────────────────────────────────────────────────────────────────────

def draw_character(surf, alpha_surf, pose, is_agent, action, anim_t,
                   hp_ratio, step, is_hit, unconscious,
                   particles):
    if is_agent:
        sc,sd,sl = (35,90,200),(20,55,140),(60,120,230)
        pc,pd    = (35,45,65),(20,25,40)
        shoe_c   = (50,50,62)
        hair_c   = (100,70,40)
        eye_c    = (60,110,200)
        fr       = True
    else:
        sc,sd,sl = (200,45,45),(140,25,25),(230,75,75)
        pc,pd    = (55,35,40),(20,25,40)
        shoe_c   = (62,40,40)
        hair_c   = (28,22,22)
        eye_c    = (110,65,35)
        fr       = False

    is_attacking = (action is not None and anim_t <= 1.05)
    has_weapon   = (action == 2 and is_attacking)

    # convenience: unpack joints
    p = pose
    cx = int(p['cx']) if 'cx' in p else int(p['head'][0])
    foot_y = (p['foot_l'][1]+p['foot_r'][1])/2

    # select back/front limbs based on facing direction
    if is_agent:
        bl_hip,bl_kn,bl_ft = p['hip_l'],p['knee_l'],p['foot_l']
        fl_hip,fl_kn,fl_ft = p['hip_r'],p['knee_r'],p['foot_r']
        ba_sh,ba_el,ba_hd  = p['shoulder_l'],p['elbow_l'],p['hand_l']
        fa_sh,fa_el,fa_hd  = p['shoulder_r'],p['elbow_r'],p['hand_r']
    else:
        bl_hip,bl_kn,bl_ft = p['hip_r'],p['knee_r'],p['foot_r']
        fl_hip,fl_kn,fl_ft = p['hip_l'],p['knee_l'],p['foot_l']
        ba_sh,ba_el,ba_hd  = p['shoulder_r'],p['elbow_r'],p['hand_r']
        fa_sh,fa_el,fa_hd  = p['shoulder_l'],p['elbow_l'],p['hand_l']

    # shadow
    draw_shadow_ellipse(surf, cx, foot_y, alpha_surf)

    # back leg
    draw_pants(surf, pc, pd, bl_hip, bl_kn, bl_ft, fl_hip, fl_kn, fl_ft)
    draw_shoe(surf, bl_ft, fr, shoe_c)

    # back arm (darker / in shadow)
    sc_back = lerp_c(sc,(0,0,0),0.28)
    sd_back = lerp_c(sd,(0,0,0),0.2)
    sl_back = lerp_c(sl,(0,0,0),0.2)
    draw_arm(surf, ba_sh, ba_el, ba_hd,
             SKIN_M, SKIN_S, sc_back, sd_back, sl_back, False)

    # torso
    torso_cy = int((p['shoulder_l'][1]+p['hip_l'][1])//2 + 2)
    draw_torso(surf, cx, torso_cy, fr, sc, sd, sl, is_hit)

    # front shoe
    draw_shoe(surf, fl_ft, fr, shoe_c)

    # front arm
    draw_arm(surf, fa_sh, fa_el, fa_hd,
             SKIN, SKIN_M, sc, sd, sl, is_attacking)

    # weapon
    if has_weapon:
        draw_weapon(surf, fa_hd, fr)

    # head
    mouth_open = is_attacking and anim_t < 0.5
    draw_head(surf, p['head'][0], p['head'][1], fr, hair_c, eye_c,
              is_hit, mouth_open, action is not None, unconscious)

    # low hp sweat
    if hp_ratio < 0.35 and not unconscious:
        particles.sweat(int(p['head'][0]), int(p['head'][1])-20)


# ──────────────────────────────────────────────────────────────────────────────
#  HUD HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def draw_hp_bar(surf, x, y, ratio, display_ratio, label, main_c, flash, fonts,
                bar_w=210, bar_h=22):
    font, small_font, _, tiny_font = fonts
    # panel bg
    panel = pygame.Surface((bar_w+22, bar_h+42), pygame.SRCALPHA)
    panel.fill((14,11,24,175))
    pygame.draw.rect(panel,(75,65,95,200),(0,0,bar_w+22,bar_h+42),1,border_radius=6)
    surf.blit(panel,(x-6,y-30))

    lbl = small_font.render(label, True,(200,200,222))
    surf.blit(lbl,(x,y-24))
    pct = tiny_font.render(f"{int(ratio*100)}%", True,(255,255,255))
    surf.blit(pct,(x+bar_w+8,y+3))

    bg = pygame.Rect(x,y,bar_w,bar_h)
    pygame.draw.rect(surf,(28,26,36),bg,border_radius=5)
    pygame.draw.rect(surf,(55,50,68),bg,1,border_radius=5)

    # drain (yellow lag)
    dw = int((bar_w-4)*max(0,display_ratio))
    if dw>0:
        pygame.draw.rect(surf,(195,175,28),pygame.Rect(x+2,y+2,dw,bar_h-4),border_radius=4)

    # hp fill
    fw = int((bar_w-4)*max(0,ratio))
    if fw>0:
        pygame.draw.rect(surf,main_c,pygame.Rect(x+2,y+2,fw,bar_h-4),border_radius=4)
        hl = pygame.Surface((fw,(bar_h-4)//3),pygame.SRCALPHA)
        hl.fill((255,255,255,38))
        surf.blit(hl,(x+2,y+2))

    # segment ticks
    for i in range(1,10):
        sx = x+2+int((bar_w-4)*i/10)
        tc = (48,44,58) if i!=5 else (68,62,82)
        pygame.draw.line(surf,tc,(sx,y+2),(sx,y+bar_h-2),1)

    # glow border when flashing
    if flash:
        gs = pygame.Surface((bar_w+10,bar_h+10),pygame.SRCALPHA)
        pygame.draw.rect(gs,(*main_c,95),(0,0,bar_w+10,bar_h+10),2,border_radius=7)
        surf.blit(gs,(x-5,y-5))

    pygame.draw.rect(surf,(115,110,135),bg,1,border_radius=5)


def draw_combo(surf, count, cx, cy, fonts):
    if count < 2:
        return
    _, _, big_font, _ = fonts
    c = (255,220,50) if count < 5 else (255,100,50) if count < 10 else (200,50,255)
    t  = big_font.render(f"x{count}", True, c)
    sh = big_font.render(f"x{count}", True,(60,35,8))
    surf.blit(sh,(cx-t.get_width()//2+2,cy+2))
    surf.blit(t, (cx-t.get_width()//2,   cy))
    _, sf, _, _ = fonts
    sub = sf.render("COMBO!", True,(255,200,80))
    surf.blit(sub,(cx-sub.get_width()//2, cy+42))


def draw_edge_flash(surf, side, intensity, w, h):
    if intensity <= 0:
        return
    flash = pygame.Surface((w,h), pygame.SRCALPHA)
    a = int(90*intensity)
    bw = 130
    if side == 'left':
        pygame.draw.rect(flash,(255,55,55,a),(0,0,bw,h))
    else:
        pygame.draw.rect(flash,(255,55,55,a),(w-bw,0,bw,h))
    surf.blit(flash,(0,0))


def draw_starburst(surf, x, y, step, alpha_base=255):
    """Old-style impact cross."""
    if step >= 8:
        return
    imp = pygame.Surface((surf.get_width(),surf.get_height()), pygame.SRCALPHA)
    a   = int(alpha_base * (1 - step/8))
    for i in range(8):
        angle  = (2*PI/8)*i + step*0.3
        length = 14 + step*6
        ex = x + math.cos(angle)*length
        ey = y + math.sin(angle)*length
        pygame.draw.line(imp,(255,255,100,a),(x,y),(int(ex),int(ey)),3)
    r = max(1, 10-step)
    pygame.draw.circle(imp,(255,255,200,a),(x,y),r)
    surf.blit(imp,(0,0))


# ──────────────────────────────────────────────────────────────────────────────
#  PRE-RENDER BACKGROUND
# ──────────────────────────────────────────────────────────────────────────────

def build_bg(w, h):
    BG_TOP=(18,15,28); BG_MID=(28,22,42); BG_BOT=(38,30,52)
    FLOOR_C=(70,60,55); FLOOR_L=(86,76,68); FLOOR_S=(50,42,38)
    surf = pygame.Surface((w,h))
    for y in range(h):
        t = y/h
        if t < 0.5:
            t2 = t*2
            c = lerp_c(BG_TOP,BG_MID,t2)
        else:
            t2 = (t-0.5)*2
            c = lerp_c(BG_MID,BG_BOT,t2)
        pygame.draw.line(surf,c,(0,y),(w,y))

    floor_top = 420
    floor_pts = [(30,floor_top),(w-30,floor_top),(w+80,h),(-80,h)]
    pygame.draw.polygon(surf,FLOOR_C,floor_pts)
    for i in range(13):
        t = i/12
        y = int(floor_top+(h-floor_top)*t)
        lx= int(30 + (-80-30)*t)
        rx= int((w-30)+(w+80-(w-30))*t)
        pygame.draw.line(surf, FLOOR_L if i%2==0 else FLOOR_S,(lx,y),(rx,y),1)
    for i in range(11):
        t = i/10
        x = int(30+(w-60)*t)
        pygame.draw.line(surf,FLOOR_S,(x,floor_top),(int(x+(t-0.5)*170),h),1)
    pygame.draw.line(surf,(110,100,90),(30,floor_top),(w-30,floor_top),2)

    # spotlights
    sp = pygame.Surface((w,h),pygame.SRCALPHA)
    pygame.draw.ellipse(sp,(255,240,200,11),(90,40,320,460))
    pygame.draw.ellipse(sp,(200,220,255,11),(490,40,320,460))
    surf.blit(sp,(0,0))
    return surf, floor_top


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN show_fight FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def show_fight(env, agent, delay=1, save_video=True, video_filename="fight_video.mp4"):
    pygame.init()
    W, H = 900, 650
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("RL Fight – Enhanced Edition")
    clock = pygame.time.Clock()

    font       = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)
    big_font   = pygame.font.Font(None, 72)
    tiny_font  = pygame.font.Font(None, 20)
    fonts = (font, small_font, big_font, tiny_font)

    action_names = {0:"PUNCH", 1:"KICK", 2:"WEAPON"}

    # ── systems ───────────────────────────────────────────────────────────────
    ptcl  = ParticleSystem(W, H)
    shake = ScreenShake()
    hp_a  = SmoothHP(env.max_hp)
    hp_o  = SmoothHP(env.max_hp)

    bg_surf, floor_top_y = build_bg(W, H)
    game_surf  = pygame.Surface((W, H))
    alpha_surf = pygame.Surface((W, H), pygame.SRCALPHA)   # reused scratch surface

    # ── state vars ────────────────────────────────────────────────────────────
    state  = env.reset()
    done   = False
    total_reward   = 0
    step           = 0
    agent_anim_t   = 2.0
    opp_anim_t     = 2.0
    last_a_action  = None
    last_o_action  = None
    agent_is_hit   = False
    opp_is_hit     = False
    flash_a        = 0.0
    flash_o        = 0.0
    last_dmg       = 0
    combo_count    = 0
    combo_timer    = 0
    agent_dead     = False
    opp_dead       = False
    death_t_a      = 0.0
    death_t_o      = 0.0
    game_over      = False
    game_over_timer= 0
    result_text    = ""
    result_color   = (255,255,255)
    impact_pos     = None
    impact_step    = 99
    damage_numbers = []   # {x,y,dx,dy,text,color,age,max_age}

    frames = []
    if save_video:
        os.makedirs("temp_frames", exist_ok=True)

    running = True
    while (not done or game_over_timer < 100) and running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False; break

        # ── step env ──────────────────────────────────────────────────────────
        if not done:
            action        = agent.act(state, eval_mode=True)
            prev_a_hp     = env.agent_hp
            prev_o_hp     = env.opponent_hp
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state         = next_state
            step         += 1

            agent_is_hit  = env.agent_hp   < prev_a_hp
            opp_is_hit    = env.opponent_hp < prev_o_hp
            last_dmg      = prev_o_hp - env.opponent_hp

            # trigger animations
            if action is not None:
                agent_anim_t = 0.0
            opp_action = getattr(env, 'last_opponent_action', None)
            if opp_action is not None and opp_action != last_o_action:
                opp_anim_t = 0.0
            last_a_action = action
            last_o_action = opp_action

            # combo
            if opp_is_hit:
                combo_count = combo_count+1 if combo_timer>0 else 1
                combo_timer = 48
            
            # ── hit effects ───────────────────────────────────────────────────
            if opp_is_hit and last_dmg > 0:
                px,py = 660, 298
                ptcl.sparks(px, py, n=20, weapon=(action==2))
                ptcl.blood(px, py, n=int(last_dmg*1.6+6))
                ptcl.stars(px, py)
                if action==2: ptcl.smoke(px, py)
                if last_dmg >= 15:
                    ptcl.shockwave(px, py); shake.trigger(9)
                else:
                    shake.trigger(4)
                flash_o = 1.0
                impact_pos, impact_step = (px,py), 0
                damage_numbers.append(dict(
                    x=px+random.randint(-14,14), y=py-30,
                    dx=random.uniform(-0.4,0.4), dy=-2.8,
                    text=f"-{last_dmg}", color=(255,50,50),
                    age=0, max_age=42))

            if agent_is_hit:
                px,py = 240, 298
                dmg = prev_a_hp - env.agent_hp
                ptcl.sparks(px, py, n=14)
                ptcl.blood(px, py, n=int(dmg*1.3+4))
                if dmg >= 15:
                    ptcl.shockwave(px, py); shake.trigger(8)
                else:
                    shake.trigger(3)
                flash_a = 1.0
                impact_pos, impact_step = (px,py), 0
                damage_numbers.append(dict(
                    x=px+random.randint(-14,14), y=py-30,
                    dx=random.uniform(-0.4,0.4), dy=-2.8,
                    text=f"-{dmg}", color=(255,100,100),
                    age=0, max_age=42))

            # footstep dust every ~12 frames
            if step % 12 == 0:
                ptcl.dust(210, floor_top_y, direction=1,  n=4)
                ptcl.dust(690, floor_top_y, direction=-1, n=4)

            # death triggers
            if env.agent_hp <= 0 and not agent_dead:
                agent_dead = True
                shake.trigger(13)
                ptcl.shockwave(210,350); ptcl.shockwave(210,350)
                ptcl.blood(210,300,32)
            if env.opponent_hp <= 0 and not opp_dead:
                opp_dead = True
                shake.trigger(13)
                ptcl.shockwave(690,350); ptcl.shockwave(690,350)
                ptcl.blood(690,300,32)

            if done and not game_over:
                game_over = True
                result_text  = "AGENT DEFEATED"  if env.agent_hp<=0 else "AGENT VICTORIOUS!"
                result_color = (255,80,80)        if env.agent_hp<=0 else (80,255,130)

        # ── advance timers ────────────────────────────────────────────────────
        agent_anim_t += 0.13
        opp_anim_t   += 0.13
        if combo_timer > 0: combo_timer -= 1
        if combo_timer == 0: combo_count = 0
        flash_a = max(0.0, flash_a - 0.14)
        flash_o = max(0.0, flash_o - 0.14)
        if agent_dead: death_t_a = min(1.0, death_t_a + 0.028)
        if opp_dead:   death_t_o = min(1.0, death_t_o + 0.028)
        impact_step += 1
        if game_over:  game_over_timer += 1

        # smooth HP
        hp_a.update(env.agent_hp)
        hp_o.update(env.opponent_hp)

        # particles / shake
        ptcl.update()
        shake_off = shake.update()

        # damage numbers
        live_dn = []
        for dn in damage_numbers:
            dn['age'] += 1; dn['x'] += dn['dx']; dn['y'] += dn['dy']
            dn['dy'] *= 0.84
            if dn['age'] < dn['max_age']: live_dn.append(dn)
        damage_numbers = live_dn

        # ── draw ──────────────────────────────────────────────────────────────
        game_surf.blit(bg_surf, (0,0))

        # poses
        if agent_dead:
            ap = get_death_pose(True, death_t_a)
        else:
            ap = get_pose(last_a_action if agent_anim_t<=1.1 else None,
                          True, step, agent_anim_t)
        if opp_dead:
            op = get_death_pose(False, death_t_o)
        else:
            op = get_pose(last_o_action if opp_anim_t<=1.1 else None,
                          False, step, opp_anim_t)

        # characters: opponent drawn first (behind)
        draw_character(game_surf, alpha_surf, op, False,
                       last_o_action if opp_anim_t<=1.1 else None, opp_anim_t,
                       env.opponent_hp/env.max_hp, step, opp_is_hit, opp_dead, ptcl)
        draw_character(game_surf, alpha_surf, ap, True,
                       last_a_action if agent_anim_t<=1.1 else None, agent_anim_t,
                       env.agent_hp/env.max_hp, step, agent_is_hit, agent_dead, ptcl)

        # particles
        ptcl.draw(game_surf)

        # starburst impact
        if impact_pos and impact_step < 9:
            draw_starburst(game_surf, *impact_pos, impact_step)

        # edge flash
        draw_edge_flash(game_surf, 'left',  flash_a, W, H)
        draw_edge_flash(game_surf, 'right', flash_o, W, H)

        # floating damage numbers
        for dn in damage_numbers:
            frac  = 1 - dn['age']/dn['max_age']
            alpha = int(255*frac)
            df    = big_font if frac > 0.6 else font
            sh = df.render(dn['text'], True,(40,0,0))
            ts = df.render(dn['text'], True, dn['color'])
            sh.set_alpha(alpha//2); ts.set_alpha(alpha)
            game_surf.blit(sh,(int(dn['x'])+2, int(dn['y'])+2))
            game_surf.blit(ts,(int(dn['x']),   int(dn['y'])))

        # VS label
        vs = big_font.render("VS", True,(155,155,55))
        wobble = math.sin(step*0.1)*2
        game_surf.blit(vs,(W//2-vs.get_width()//2, int(290+wobble)))

        # combo
        if combo_count >= 2:
            draw_combo(game_surf, combo_count, W//2, 340, fonts)

        # ── HUD bars ──────────────────────────────────────────────────────────
        a_ratio = env.agent_hp    / env.max_hp
        o_ratio = env.opponent_hp / env.max_hp
        def hp_color(r):
            if r>0.5: return (50,200,60)
            if r>0.2: return (220,165,25)
            return (220,50,50)

        draw_hp_bar(game_surf, 20, 40, a_ratio, hp_a.display,
                    "AGENT", hp_color(a_ratio), hp_a.flash_timer>0, fonts)
        draw_hp_bar(game_surf, W-238, 40, o_ratio, hp_o.display,
                    "OPPONENT", hp_color(o_ratio), hp_o.flash_timer>0, fonts)

        # action labels
        if last_a_action is not None and agent_anim_t < 1.6:
            at = small_font.render(f"▶ {action_names.get(last_a_action,'?')}", True,(80,180,255))
            game_surf.blit(at,(20,80))
        if last_o_action is not None and opp_anim_t < 1.6:
            ot = small_font.render(f"{action_names.get(last_o_action,'?')} ◀", True,(255,130,130))
            game_surf.blit(ot,(W-20-ot.get_width(), 80))

        # title bar
        tb = pygame.Surface((280,28), pygame.SRCALPHA); tb.fill((14,11,24,155))
        game_surf.blit(tb,(W//2-140,0))
        tt = font.render("RL FIGHTING ARENA", True,(200,200,228))
        game_surf.blit(tt,(W//2-tt.get_width()//2, 2))

        # footer
        fb = pygame.Surface((W,24),pygame.SRCALPHA); fb.fill((0,0,0,95))
        game_surf.blit(fb,(0,H-24))
        ft = small_font.render(f"Step: {step}   Reward: {total_reward:.1f}", True,(155,210,155))
        game_surf.blit(ft,(W//2-ft.get_width()//2, H-22))

        # game-over overlay
        if game_over:
            t_fade = min(1.0, game_over_timer/35)
            ov = pygame.Surface((W,H),pygame.SRCALPHA)
            ov.fill((0,0,0,int(155*t_fade)))
            game_surf.blit(ov,(0,0))
            if t_fade > 0.45:
                ry  = H//2-55
                shd = big_font.render(result_text, True,(0,0,0))
                res = big_font.render(result_text, True, result_color)
                game_surf.blit(shd,(W//2-res.get_width()//2+3,ry+3))
                game_surf.blit(res,(W//2-res.get_width()//2,   ry))
                sub = font.render(f"Steps: {step}   Total Reward: {total_reward:.1f}",
                                  True,(200,200,200))
                game_surf.blit(sub,(W//2-sub.get_width()//2, ry+72))

        # apply shake
        screen.fill((0,0,0))
        screen.blit(game_surf, shake_off)
        pygame.display.flip()

        if save_video:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame,(1,0,2))
            frames.append(frame)

        target_fps = max(1, int(1/delay)) if delay > 0 else 60
        clock.tick(target_fps)

        opp_is_hit = agent_is_hit = False   # reset per-frame hit flags

    # hold end screen
    if save_video and frames:
        hold = int(3/delay) if delay>0 else 90
        for _ in range(hold):
            frames.append(frames[-1])

    if save_video and frames:
        print(f"Saving video → {video_filename}")
        fps = max(1, int(1/delay)) if delay>0 else 30
        writer = imageio.get_writer(video_filename, fps=fps)
        for fr in frames:
            writer.append_data(fr)
        writer.close()
        print("Saved.")

    pygame.time.wait(2000)
    pygame.quit()
    return frames


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from environment import FightingEnv
    import random

    class RandomAgent:
        def act(self, state, eval_mode=False):
            return random.choice([0, 1, 2])

    env   = FightingEnv()
    agent = RandomAgent()
    show_fight(env, agent, delay=0.15, save_video=True,
               video_filename="test_fight_enhanced.mp4")