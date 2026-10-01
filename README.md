# Signcall

Signcall is a Next.js video calling starter that uses browser-native WebRTC for peer-to-peer media and MongoDB for room presence and signaling. It does not use Agora or another managed media provider.

## Setup

Create `.env.local` in the project root:

```dotenv
MONGODB_URI=mongodb://localhost:27017/sign-language
SESSION_SECRET=replace_with_at_least_32_random_characters
```

Start MongoDB, then run:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`, sign in or create an account, then select **Create a room**. Share the invite link; the other participant can paste it into the **Room ID or invite link** field or open it directly.

Camera and microphone access requires localhost or HTTPS and browser permission. WebRTC media is sent peer-to-peer. Signaling messages and room presence are stored temporarily in MongoDB and removed using TTL indexes. A public STUN server helps peers discover direct routes, but some networks block direct peer connections. For reliable connectivity across restrictive NATs/firewalls, configure a TURN relay such as a self-hosted coturn server and add it to the `RTCPeerConnection` ICE server list.

## API

- `POST /api/auth/register` creates a normal account; role is assigned on the server.
- `POST /api/auth/login` verifies credentials and sets a signed, HTTP-only session cookie.
- `GET /api/auth/me` returns the current account when a valid session is present.
- `POST /api/auth/logout` clears the session cookie.
- `GET`, `POST`, and `DELETE /api/signaling/[roomId]` heartbeat room presence, exchange WebRTC offer/answer/ICE messages, and leave a room.

Room URLs are unlisted links, not access-controlled invitations. The initial signaling endpoint does not require a signed-in session, and this prototype supports one-to-one calls. Add server-verified room membership and rate limits before using it as a private production service.
