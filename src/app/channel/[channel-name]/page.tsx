import Call from "@/components/Call";
import Link from "next/link";

interface PageProps {
  params: Promise<Record<string, string>>;
}

export default async function ChannelPage({ params }: PageProps) {
  const routeParams = await params;
  const channelName = routeParams.channelName ?? routeParams["channel-name"];

  if (!channelName || !/^[a-zA-Z0-9_-]{1,64}$/.test(channelName)) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#f4f7f6] px-6 text-center">
        <div className="max-w-md rounded-3xl border border-slate-200 bg-white p-8 shadow-xl">
          <h1 className="text-xl font-semibold text-slate-900">That room link isn’t valid</h1>
          <p className="mt-2 text-sm leading-6 text-slate-500">Room links can contain up to 64 letters, numbers, dashes or underscores.</p>
          <Link href="/" className="mt-6 inline-flex min-h-11 items-center rounded-xl bg-teal-700 px-5 font-semibold text-white hover:bg-teal-800">Back to home</Link>
        </div>
      </div>
    );
  }

  return <Call channelName={channelName} />;
}
