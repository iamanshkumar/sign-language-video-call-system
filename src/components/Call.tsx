"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";

type SignalType = "offer" | "answer" | "ice" | "leave";
interface SignalMessage {
  id: string;
  fromClientId: string;
  type: SignalType;
  payload: unknown;
}
interface PresenceResponse {
  participants: string[];
  messages: SignalMessage[];
  error?: string;
}

function Icon({ name, className = "" }: React.SVGProps<SVGSVGElement> & { name: "mic" | "micOff" | "camera" | "cameraOff" | "phone" | "copy" | "users" | "spinner" }) {
  const paths: Record<typeof name, React.ReactNode> = {
    mic: <><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-4 0h8"/></>,
    micOff: <><path d="m3 3 18 18M9 9v3a3 3 0 0 0 5.1 2.1M15 9V5a3 3 0 0 0-5.9-.7M5 10v2a7 7 0 0 0 11.8 5M19 10v2m-7 7v3m-4 0h8"/></>,
    camera: <><rect x="3" y="6" width="13" height="12" rx="2"/><path d="m16 10 5-3v10l-5-3z"/></>,
    cameraOff: <><path d="m3 3 18 18M10 6h4a2 2 0 0 1 2 2v2l5-3v10l-5-3v2M3 8a2 2 0 0 1 2-2h1m-3 4v6a2 2 0 0 0 2 2h9"/></>,
    phone: <><path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.4 19.4 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7l.5 3.1a2 2 0 0 1-.6 1.7L7.1 10.4a16 16 0 0 0 6 6l1.9-1.9a2 2 0 0 1 1.7-.6l3.1.5a2 2 0 0 1 2.2 2.5z"/></>,
    copy: <><rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>,
    users: <><path d="M16 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2m6-10a4 4 0 1 0 0-8 4 4 0 0 0 0 8zm10 10v-2a4 4 0 0 0-3-3.9m-1-12.1a4 4 0 0 1 0 7.8"/></>,
    spinner: <><path d="M21 12a9 9 0 1 1-6.2-8.6"/></>,
  };
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">{paths[name]}</svg>;
}

export default function Call({ channelName }: { channelName: string }) {
  const router = useRouter();
  const [clientId] = useState(() => crypto.randomUUID());
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [participants, setParticipants] = useState<string[]>([]);
  const [callStatus, setCallStatus] = useState("Starting camera…");
  const [error, setError] = useState("");
  const [micOn, setMicOn] = useState(true);
  const [cameraOn, setCameraOn] = useState(true);
  const [copied, setCopied] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const peerRef = useRef<{ clientId: string; connection: RTCPeerConnection } | null>(null);
  const pendingIceRef = useRef<RTCIceCandidateInit[]>([]);
  const cursorRef = useRef("");
  const pollBusyRef = useRef(false);
  const offeredToRef = useRef("");
  const callStartedAtRef = useRef<number | null>(null);
  const signalingUrl = useMemo(() => `/api/signaling/${encodeURIComponent(channelName)}`, [channelName]);

  useEffect(() => {
    let cancelled = false;
    const mediaRequest = navigator.mediaDevices?.getUserMedia
      ? navigator.mediaDevices.getUserMedia({
          audio: true,
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        })
      : Promise.reject(new Error("This browser cannot access a camera. Open the app on localhost or over HTTPS."));

    mediaRequest.then((stream) => {
      if (cancelled) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      localStreamRef.current = stream;
      setLocalStream(stream);
      setError("");
      setCallStatus("Waiting for someone to join");
    }).catch((mediaError: unknown) => {
      const message = mediaError instanceof Error ? mediaError.message : "Camera or microphone permission was denied.";
      setError(`Could not access your camera or microphone: ${message}`);
      setCallStatus("Camera unavailable");
    });

    return () => {
      cancelled = true;
      localStreamRef.current?.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (localVideoRef.current) localVideoRef.current.srcObject = localStream;
  }, [localStream]);
  useEffect(() => {
    if (remoteVideoRef.current) remoteVideoRef.current.srcObject = remoteStream;
  }, [remoteStream]);

  const sendSignal = useCallback(async (type: SignalType, toClientId: string, payload: object = {}) => {
    if (!clientId) return;
    const response = await fetch(signalingUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fromClientId: clientId, toClientId, type, payload }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || "Could not send a call signal.");
    }
  }, [clientId, signalingUrl]);

  const ensurePeer = useCallback((remoteClientId: string) => {
    if (peerRef.current?.clientId === remoteClientId && peerRef.current.connection.signalingState !== "closed") {
      return peerRef.current.connection;
    }

    peerRef.current?.connection.close();
    pendingIceRef.current = [];
    const connection = new RTCPeerConnection({
      iceServers: [
        { urls: "stun:stun.l.google.com:19302" },
        { urls: "stun:stun1.l.google.com:19302" },
      ],
    });
    localStreamRef.current?.getTracks().forEach((track) => connection.addTrack(track, localStreamRef.current!));
    connection.ontrack = (event) => {
      const stream = event.streams[0] ?? new MediaStream([event.track]);
      setRemoteStream(stream);
    };
    connection.onicecandidate = (event) => {
      if (!event.candidate) return;
      void sendSignal("ice", remoteClientId, { ...event.candidate.toJSON() }).catch((signalError: unknown) => {
        setError(signalError instanceof Error ? signalError.message : "Could not send network connection details.");
      });
    };
    connection.onconnectionstatechange = () => {
      if (connection.connectionState === "connected") {
        setError("");
        setCallStatus("Connected");
        if (!callStartedAtRef.current) callStartedAtRef.current = Date.now();
      } else if (connection.connectionState === "failed") {
        setCallStatus("Could not connect");
        setError("A direct peer connection could not be made on these networks. This can happen behind restrictive routers; a TURN relay is needed for reliable fallback.");
      } else if (connection.connectionState === "connecting") {
        setCallStatus("Connecting to participant…");
      } else if (connection.connectionState === "disconnected") {
        setCallStatus("Reconnecting…");
      }
    };
    peerRef.current = { clientId: remoteClientId, connection };
    setCallStatus("Connecting to participant…");
    return connection;
  }, [sendSignal]);

  const applySignal = useCallback(async (message: SignalMessage) => {
    if (message.type === "leave") {
      if (peerRef.current?.clientId === message.fromClientId) {
        peerRef.current.connection.close();
        peerRef.current = null;
        pendingIceRef.current = [];
        offeredToRef.current = "";
        setRemoteStream(null);
        setCallStatus("Waiting for someone to join");
      }
      return;
    }

    const connection = ensurePeer(message.fromClientId);
    if (message.type === "ice") {
      const candidate = message.payload as RTCIceCandidateInit;
      if (connection.remoteDescription) await connection.addIceCandidate(candidate);
      else pendingIceRef.current.push(candidate);
      return;
    }

    if (message.type === "offer") {
      await connection.setRemoteDescription(message.payload as unknown as RTCSessionDescriptionInit);
      for (const candidate of pendingIceRef.current.splice(0)) await connection.addIceCandidate(candidate);
      const answer = await connection.createAnswer();
      await connection.setLocalDescription(answer);
      if (connection.localDescription) {
        await sendSignal("answer", message.fromClientId, { ...connection.localDescription.toJSON() });
      }
      return;
    }

    if (message.type === "answer" && !connection.remoteDescription) {
      await connection.setRemoteDescription(message.payload as unknown as RTCSessionDescriptionInit);
      for (const candidate of pendingIceRef.current.splice(0)) await connection.addIceCandidate(candidate);
    }
  }, [ensurePeer, sendSignal]);

  useEffect(() => {
    if (!clientId || !localStream) return;
    let stopped = false;
    const poll = async () => {
      if (stopped || pollBusyRef.current) return;
      pollBusyRef.current = true;
      try {
        const query = new URLSearchParams({ clientId });
        if (cursorRef.current) query.set("after", cursorRef.current);
        const response = await fetch(`${signalingUrl}?${query}`, { cache: "no-store" });
        const data = await response.json() as PresenceResponse;
        if (!response.ok) throw new Error(data.error || "Could not connect to room signaling.");
        if (stopped) return;

        setError("");
        setParticipants(data.participants);
        const otherPeers = data.participants.filter((participantId) => participantId !== clientId);
        const remoteClientId = otherPeers.sort()[0];
        if (otherPeers.length > 1) {
          setError("This free peer-to-peer room supports one other participant at a time.");
        }

        if (peerRef.current && !otherPeers.includes(peerRef.current.clientId)) {
          peerRef.current.connection.close();
          peerRef.current = null;
          pendingIceRef.current = [];
          offeredToRef.current = "";
          setRemoteStream(null);
          setCallStatus("Waiting for someone to join");
        }

        if (remoteClientId) {
          const connection = ensurePeer(remoteClientId);
          if (clientId.localeCompare(remoteClientId) < 0 && offeredToRef.current !== remoteClientId && connection.signalingState === "stable") {
            offeredToRef.current = remoteClientId;
            const offer = await connection.createOffer();
            await connection.setLocalDescription(offer);
            if (connection.localDescription) {
              await sendSignal("offer", remoteClientId, { ...connection.localDescription.toJSON() });
            }
          }
        } else if (!peerRef.current) {
          setCallStatus("Waiting for someone to join");
        }

        for (const message of data.messages) {
          if (stopped) break;
          await applySignal(message);
          cursorRef.current = message.id;
        }
      } catch (pollError) {
        if (!stopped) {
          setCallStatus("Room signaling unavailable");
          setError(pollError instanceof Error ? pollError.message : "Room signaling is unavailable.");
        }
      } finally {
        pollBusyRef.current = false;
      }
    };

    void poll();
    const timer = window.setInterval(() => void poll(), 1000);
    return () => {
      stopped = true;
      window.clearInterval(timer);
      const peer = peerRef.current;
      peer?.connection.close();
      peerRef.current = null;
      void fetch(`${signalingUrl}?clientId=${encodeURIComponent(clientId)}`, { method: "DELETE", keepalive: true });
    };
  }, [applySignal, clientId, ensurePeer, localStream, sendSignal, signalingUrl]);

  useEffect(() => {
    if (!callStartedAtRef.current) return;
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - callStartedAtRef.current!) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [callStatus]);

  async function endCall() {
    if (clientId) {
      await fetch(`${signalingUrl}?clientId=${encodeURIComponent(clientId)}`, { method: "DELETE" }).catch(() => undefined);
    }
    peerRef.current?.connection.close();
    peerRef.current = null;
    localStreamRef.current?.getTracks().forEach((track) => track.stop());
    router.push("/");
  }

  async function copyInvite() {
    try {
      await navigator.clipboard.writeText(`${window.location.origin}/channel/${encodeURIComponent(channelName)}`);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setError("Clipboard access is unavailable. Copy the room URL from the address bar.");
    }
  }

  function toggleMicrophone() {
    const next = !micOn;
    localStreamRef.current?.getAudioTracks().forEach((track) => { track.enabled = next; });
    setMicOn(next);
  }

  function toggleCamera() {
    const next = !cameraOn;
    localStreamRef.current?.getVideoTracks().forEach((track) => { track.enabled = next; });
    setCameraOn(next);
  }

  const duration = useMemo(() => `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(elapsed % 60).padStart(2, "0")}`, [elapsed]);
  const otherParticipants = participants.filter((participantId) => participantId !== clientId);

  return (
    <main className="flex min-h-screen flex-col bg-[#f4f7f6] text-slate-900">
      <header className="flex h-[76px] items-center justify-between border-b border-slate-200/80 bg-white px-5 sm:px-8">
        <Link href="/" className="flex items-center gap-3"><div className="flex h-10 w-10 items-center justify-center rounded-[14px] bg-teal-700 text-white"><span className="text-lg">↗</span></div><span className="text-lg font-bold tracking-tight">signcall<span className="text-teal-700">.</span></span></Link>
        <div className="flex items-center gap-3"><div className="hidden items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600 sm:flex"><span className={`h-2 w-2 rounded-full ${callStatus === "Connected" ? "bg-emerald-500" : error ? "bg-rose-500" : "animate-pulse bg-amber-400"}`} />{callStatus}</div><span className="font-mono text-sm font-medium text-slate-500">{duration}</span></div>
      </header>

      <section className="mx-auto flex w-full max-w-[1500px] flex-1 flex-col px-4 py-5 sm:px-7 sm:py-7">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <div><div className="flex items-center gap-2"><h1 className="text-xl font-semibold tracking-tight">{channelName}</h1><span className="rounded-full bg-teal-50 px-2.5 py-1 text-[11px] font-semibold text-teal-800">ROOM</span></div><p className="mt-1 text-sm text-slate-500">A quiet space for a clear conversation.</p></div>
          <div className="flex items-center gap-2"><div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-600"><Icon name="users" className="h-4 w-4"/><span>{otherParticipants.length + 1}</span></div><Button variant="outline" onClick={copyInvite}><Icon name="copy" className="h-4 w-4"/>{copied ? "Copied" : "Invite"}</Button></div>
        </div>

        {error && <div role="alert" className="mb-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm leading-6 text-rose-800">{error}</div>}

        <div className={`grid flex-1 content-center gap-4 ${otherParticipants.length ? "grid-cols-1 xl:grid-cols-2" : "grid-cols-1"}`}>
          <div className={`group relative overflow-hidden rounded-[26px] bg-[#dfe8e3] shadow-sm ${otherParticipants.length ? "min-h-[320px]" : "min-h-[390px] sm:min-h-[520px]"}`}>
            {cameraOn && localStream ? <video ref={localVideoRef} autoPlay muted playsInline className="absolute inset-0 h-full w-full object-contain" /> : <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-[#e5eee9] to-[#cbdcd3]"><div className="flex h-24 w-24 items-center justify-center rounded-full bg-white text-3xl font-semibold text-teal-800 shadow-sm">Y</div><p className="mt-4 text-sm font-medium text-slate-600">{localStream ? "Your camera is off" : "Waiting for camera permission…"}</p></div>}
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-slate-950/45 via-transparent to-slate-950/10"/><div className="absolute left-4 top-4 rounded-full border border-white/20 bg-black/25 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-md">You</div><div className="absolute bottom-4 left-4 flex items-center gap-2 text-sm font-medium text-white"><span className="h-2 w-2 rounded-full bg-emerald-400"/>You</div>
            {!micOn && <div className="absolute bottom-3 right-3 rounded-full bg-black/35 p-2 text-white"><Icon name="micOff" className="h-4 w-4"/></div>}
          </div>

          {otherParticipants.length > 0 && <div className="relative min-h-[320px] overflow-hidden rounded-[26px] bg-[#dfe8e3] shadow-sm">
            {remoteStream ? <video ref={remoteVideoRef} autoPlay playsInline className="absolute inset-0 h-full w-full object-contain" /> : <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-[#e8e9f0] to-[#d5d9e7]"><div className="flex h-24 w-24 items-center justify-center rounded-full bg-white text-3xl font-semibold text-indigo-800 shadow-sm">G</div><p className="mt-4 text-sm font-medium text-slate-600">Connecting video…</p></div>}
            <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-slate-950/45 via-transparent to-slate-950/10"/><div className="absolute left-4 top-4 rounded-full border border-white/20 bg-black/25 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-md">Guest</div><div className="absolute bottom-4 left-4 flex items-center gap-2 text-sm font-medium text-white"><span className="h-2 w-2 rounded-full bg-emerald-400"/>Participant</div>
          </div>}

          {otherParticipants.length === 0 && <div className="absolute bottom-28 left-1/2 hidden -translate-x-1/2 items-center gap-2 rounded-full border border-white/60 bg-white/80 px-4 py-2 text-xs text-slate-500 shadow-sm backdrop-blur-md sm:flex"><span className="flex h-5 w-5 items-center justify-center rounded-full bg-teal-50 text-teal-700">✳</span>{callStatus === "Waiting for someone to join" ? "Share the room link when you’re ready" : callStatus}</div>}
        </div>

        <div className="mt-5 flex flex-wrap items-center justify-center gap-3 pb-1">
          <Button variant={micOn ? "outline" : "danger"} size="icon" aria-label={micOn ? "Mute microphone" : "Turn microphone on"} aria-pressed={!micOn} onClick={toggleMicrophone}><Icon name={micOn ? "mic" : "micOff"} className="h-5 w-5"/></Button>
          <Button variant={cameraOn ? "outline" : "danger"} size="icon" aria-label={cameraOn ? "Turn camera off" : "Turn camera on"} aria-pressed={!cameraOn} onClick={toggleCamera}><Icon name={cameraOn ? "camera" : "cameraOff"} className="h-5 w-5"/></Button>
          <Button variant="danger" className="rounded-full px-6" onClick={() => void endCall()}><Icon name="phone" className="h-4 w-4 rotate-[135deg]"/>Leave call</Button>
        </div>
        <p className="mt-3 text-center text-xs text-slate-400">Peer-to-peer video. Keep your hands visible and gestures in frame.</p>
      </section>
    </main>
  );
}
