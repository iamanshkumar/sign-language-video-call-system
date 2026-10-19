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

Open [http://localhost:3000](http://localhost:3000), create an account or sign in, then choose **Create a room**. The app generates a unique room ID and opens its call page. Share the invite link with the other participant, who can open it directly or paste the URL or room ID into **Room ID or invite link**.

Both people must use the same room ID and an app URL they can reach. A `localhost` invite link only works on the same computer; use a deployed HTTPS address for participants on other devices.

## How calls connect

WebRTC negotiates the peer-to-peer media connection. The app exchanges offer, answer, and ICE connection messages through `GET`, `POST`, and `DELETE /api/signaling/[roomId]`. MongoDB presence records expire after inactivity, and signaling documents have a TTL index. Camera and microphone streams remain in the browsers and are sent peer-to-peer.

The current client uses public STUN servers to help browsers find a direct network path. STUN alone cannot connect every network. For calls across restrictive NATs or firewalls, run a TURN relay such as self-hosted coturn and configure it in the `RTCPeerConnection` ICE server list in `src/components/Call.tsx`. TURN relays media and requires a publicly reachable server; the signaling database alone cannot provide that fallback.

This implementation supports one-to-one calls. Room URLs are unlisted links, not access-controlled invitations, and the signaling endpoint does not currently require a signed-in session. Add server-verified room membership and rate limits before using it as a private production service.

## Troubleshooting

- **The member count stays at 1:** the browser has not received the room presence list. Confirm MongoDB is running and `MONGODB_URI` connects to it in `.env.local`.
- **`Unexpected token '<' ... is not valid JSON`:** the browser expected a JSON response from `/api/signaling/...` but received an HTML page, commonly a Next.js 404 or error page. Stop the dev server and run `npm run dev` from this project directory so Next.js loads the signaling route. Check the server terminal for the underlying route error.
- **Room signaling unavailable:** inspect the failed `/api/signaling/...` request in the browser Network panel and the Next.js terminal. MongoDB connection failures are returned as a JSON 503 response.
- **The other participant appears but video does not connect:** confirm both browsers opened the same room link, allow camera/microphone access, and check whether the networks require a TURN relay.
- **Another device cannot open the invite:** do not use a `localhost` link across devices. Deploy the app over HTTPS and share that public app URL.

## API overview

- `POST /api/auth/register` creates an account.
- `POST /api/auth/login` verifies credentials and sets a signed, HTTP-only session cookie.
- `GET /api/auth/me` returns the current account when a valid session is present.
- `POST /api/auth/logout` clears the session cookie.
- `GET`, `POST`, and `DELETE /api/signaling/[roomId]` maintain presence, exchange WebRTC signaling messages, and leave a room.
