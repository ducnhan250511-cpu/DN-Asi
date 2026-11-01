create table if not exists conversations (
  id text primary key,
  user_id text not null,
  title text not null,
  mode text not null default 'chat',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists conversations_user_idx on conversations (user_id, updated_at desc);

create table if not exists messages (
  id text primary key,
  conversation_id text not null references conversations (id) on delete cascade,
  user_id text not null,
  role text not null,
  content text not null,
  meta text not null default '{}',
  rating smallint,
  created_at timestamptz not null default now()
);

create index if not exists messages_conv_idx on messages (conversation_id, created_at);

create table if not exists user_settings (
  user_id text primary key,
  theme text not null default 'dark',
  temperature real not null default 0.8,
  top_k integer not null default 40,
  top_p real not null default 0.9,
  max_tokens integer not null default 80,
  language text not null default 'vi'
);

create table if not exists study_cards (
  id text primary key,
  user_id text not null,
  deck text not null,
  front text not null,
  back text not null,
  box integer not null default 0,
  due_at timestamptz not null default now()
);

create index if not exists study_cards_user_idx on study_cards (user_id, due_at);

create table if not exists file_meta (
  id text primary key,
  user_id text not null,
  name text not null,
  mime text not null,
  size_bytes integer not null,
  created_at timestamptz not null default now()
);

create index if not exists file_meta_user_idx on file_meta (user_id, created_at desc);
