# Signcall

Signcall is a Next.js video calling app for one-to-one conversations. It uses browser-native WebRTC to send audio and video directly between participants. MongoDB stores short-lived room presence and signaling messages; media does not pass through MongoDB or a hosted calling provider.

## Requirements

- Node.js and npm
- MongoDB, running locally or hosted
- A modern browser with camera and microphone access

Camera and microphone access works on `localhost` during development and on HTTPS after deployment.

## Configure

Create `.env.local` in the project root:

```dotenv
MONGODB_URI=mongodb://localhost:27017/sign-language
SESSION_SECRET=replace_with_a_long_random_secret
```

Use a unique, random `SESSION_SECRET` with at least 32 characters. Keep `.env.local` private; it is ignored by Git. The old `NEXT_PUBLIC_AGORA_APP_ID` variable is no longer used.

## Run locally

Start MongoDB, then from the project directory run:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), create an account or sign in, then choose **Create a room**. The app generates a unique room ID and opens its call page. Share the invite link with the other participant, who can open it directly or paste the URL or room ID into **Room ID or invite link**. A legacy link in the form `/?room=ROOM_ID` also opens that room automatically.

Both people must use the same room ID and an app URL they can reach. A `localhost` invite link only works on the same computer; use a deployed HTTPS address for participants on other devices. For local testing over Wi-Fi, `npm run dev` listens on the network and Next allows `192.168.x.x` development origins for HMR. Browsers block camera and microphone access from a plain-HTTP LAN address, so use `localhost` on the development computer or HTTPS for another device.

## How calls connect

WebRTC negotiates the peer-to-peer media connection. The app exchanges offer, answer, and ICE connection messages through `GET`, `POST`, and `DELETE /api/signaling/[roomId]`. MongoDB presence records expire after inactivity, and signaling documents have a TTL index. Camera and microphone streams remain in the browsers and are sent peer-to-peer.

The client uses public STUN servers to find a direct network path. STUN alone cannot connect every network. For calls across restrictive NATs or firewalls, run a TURN relay such as self-hosted coturn, enable its REST-style shared-secret authentication (`use-auth-secret` and `static-auth-secret`), then set `TURN_URLS` and `TURN_SHARED_SECRET` in `.env.local`. The server issues short-lived TURN credentials through `/api/turn` (also available at `/api/rtc/ice-servers`); the shared secret is read only on the server and is never sent to browsers. TURN is a WebRTC network relay, not a hosted calling provider. For local Docker testing, see [Local TURN testing](#local-turn-testing).

## Local TURN testing

Generate a random 32-byte shared secret:

```bash
openssl rand -hex 32
```

PowerShell alternative:

```powershell
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$bytes = New-Object byte[] 32
$rng.GetBytes($bytes)
-join ($bytes | ForEach-Object { $_.ToString('x2') })
```

Copy `.env.local.example` to `.env.local` if you do not already have a local environment file. If `.env.local` already exists, preserve its MongoDB and session settings and copy in the TURN/test variables from the example. Replace `TURN_SHARED_SECRET` with the generated value, and use the same value for coturn and the Next.js server. `.env.local` is git-ignored; never commit or share the secret.

Start the local coturn container from the project root:

```bash
docker compose --env-file .env.local -f docker-compose.turn.yml up -d
```

If you change or rotate `TURN_SHARED_SECRET`, recreate coturn with the same `.env.local` value and restart Next.js so both services use matching credentials:

```bash
docker compose --env-file .env.local -f docker-compose.turn.yml down
docker compose --env-file .env.local -f docker-compose.turn.yml up -d
```

Check that it is running and inspect its logs:

```bash
docker ps --filter name=coturn
docker logs coturn
```

Restart Next.js after changing `.env.local`; Next.js reads environment variables at startup. Then open [http://localhost:3000/api/turn](http://localhost:3000/api/turn) and confirm it returns JSON with an `iceServers` array containing a `turn:localhost:3478` entry. The response includes temporary credentials, so do not post it publicly.

Set `NEXT_PUBLIC_FORCE_TURN_RELAY=true` in `.env.local`, restart `npm run dev`, then open two tabs at `http://localhost:3000`, join the same room, and start a call. In `chrome://webrtc-internals`, inspect the selected candidate pair and confirm a `relay` candidate. In development, the browser console also logs the selected local/remote candidate types. Coturn activity is visible with `docker logs coturn`. Set `NEXT_PUBLIC_FORCE_TURN_RELAY=false` and restart Next.js when the relay-only check is complete.

The Compose network gives coturn a stable container address and maps that private relay address to `TURN_EXTERNAL_IP` (default `127.0.0.1`), which Docker Desktop can forward to the host. The Compose service starts `turnserver` directly because the image entrypoint can consume the short `-n` option as an `echo` flag. The TURN relay ports are mapped without translation. This default works only when both test browsers are on the same computer; `localhost` is not reachable from a participant's other computer or phone.

### Test with someone on another computer

You can host coturn on your Mac without renting a server only if your home network accepts inbound connections and gives you a public IP. Keep the Mac awake and Docker Desktop running during the call.

1. In your router, reserve a stable LAN address for the Mac, then forward **UDP and TCP 3478** plus **UDP 49160–49200** to that Mac address. Allow these ports through the Mac firewall as well.
2. Find the router's public IPv4 address. Compare it with the WAN address shown in the router: if the router has a private address or an address in `100.64.0.0/10`, your ISP is likely using CGNAT, and ordinary port forwarding will not make coturn reachable.
3. In `.env.local`, replace the local TURN values with the public address (or a DNS name for the TURN URL):

   ```dotenv
   TURN_URLS=turn:<PUBLIC_IP>:3478?transport=udp
   TURN_EXTERNAL_IP=<PUBLIC_IP>
   ```

   Replace `<PUBLIC_IP>` with the same numeric public IPv4 address in both entries. Keep `TURN_SHARED_SECRET` unchanged and private. A DNS name can replace the IP in `TURN_URLS`, but `TURN_EXTERNAL_IP` must remain the public IP.
4. Recreate coturn and restart Next.js so both load the new values:

   ```bash
   docker compose --env-file .env.local -f docker-compose.turn.yml down
   docker compose --env-file .env.local -f docker-compose.turn.yml up -d
   npm run dev
   ```

5. Share the HTTPS app URL (for example, your dev tunnel URL) and the same room link. Test from a participant on a different network, such as cellular data. `NEXT_PUBLIC_FORCE_TURN_RELAY=true` can confirm TURN is being used; restart Next.js after changing it, then set it back to `false` after the check.

The HTTPS tunnel only carries the web app and signaling. Coturn media traffic still connects directly to the public IP over the forwarded UDP ports. If your ISP uses CGNAT, blocks inbound UDP, or your router cannot forward ports, host coturn on a publicly reachable server instead; a local Docker container cannot relay media to remote computers in that network setup. If your public IP changes, update both values and recreate coturn/restart Next.js.

This implementation supports one-to-one calls. Room URLs are unlisted links, not access-controlled invitations, and the signaling endpoint does not currently require a signed-in session. Add server-verified room membership and rate limits before using it as a private production service.

## Troubleshooting

- **The member count stays at 1:** the browser has not received the room presence list. Confirm MongoDB is running and `MONGODB_URI` connects to it in `.env.local`.
- **`Unexpected token '<' ... is not valid JSON`:** the browser expected a JSON response from `/api/signaling/...` but received an HTML page, commonly a Next.js 404 or error page. Stop the dev server and run `npm run dev` from this project directory so Next.js loads the signaling route. Check the server terminal for the underlying route error.
- **Room signaling unavailable:** inspect the failed `/api/signaling/...` request in the browser Network panel and the Next.js terminal. MongoDB connection failures are returned as a JSON 503 response.
- **The other participant appears but video does not connect:** confirm both browsers opened the same room link, allow camera/microphone access, and check whether the networks require a TURN relay.
- **No TURN relay candidate / ICE server request failed:** make sure Docker Desktop is running and `docker ps --filter name=coturn` shows the coturn container as `Up`; then inspect `docker logs coturn`. The local Compose network must advertise the address in `TURN_EXTERNAL_IP` while using coturn's fixed private container address for relay sockets. Recreate coturn after changing its network settings. For remote computers, make sure `TURN_URLS` uses a publicly reachable address and the router/firewall forwards the coturn listener and relay ports. A dev tunnel forwards the web page, not coturn's UDP media ports; `localhost` and `127.0.0.1` refer to each participant's own computer.
- **Another device cannot open the invite:** do not use a `localhost` link across devices. Deploy the app over HTTPS and share that public app URL.
- **Repeated `/_next/webpack-hmr` WebSocket errors:** restart the dev server after changing `next.config.ts`. Keep using the LAN address from the same network; if the computer's address is outside `192.168.x.x`, add its hostname to `allowedDevOrigins` in `next.config.ts`.

## API overview

- `POST /api/auth/register` creates an account.
- `POST /api/auth/login` verifies credentials and sets a signed, HTTP-only session cookie.
- `GET /api/auth/me` returns the current account when a valid session is present.
- `POST /api/auth/logout` clears the session cookie.
- `GET /api/turn` returns STUN/TURN ICE server configuration with short-lived TURN credentials when configured.
- `GET`, `POST`, and `DELETE /api/signaling/[roomId]` maintain presence, exchange WebRTC signaling messages, and leave a room.
