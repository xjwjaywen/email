import { useCallback, useEffect, useState } from 'react';
import { api } from '../services/api';
import type { Conversation, Message } from '../types';

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const loadConversations = useCallback(async () => {
    const list = await api.listConversations();
    setConversations(list);
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  const selectConversation = useCallback(async (id: string) => {
    setActiveId(id);
    const data = await api.getConversation(id);
    setMessages(data.messages || []);
  }, []);

  const createConversation = useCallback(async () => {
    const conv = await api.createConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
    setMessages([]);
    return conv;
  }, []);

  const deleteConversation = useCallback(
    async (id: string) => {
      await api.deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (activeId === id) {
        setActiveId(null);
        setMessages([]);
      }
    },
    [activeId],
  );

  const renameConversation = useCallback(async (id: string, title: string) => {
    const updated = await api.updateConversation(id, title);
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, title: updated.title } : c)));
  }, []);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!activeId || !content.trim()) return;
      setLoading(true);
      try {
        const res = await api.sendMessage(activeId, content);
        setMessages((prev) => [...prev, res.user_message, res.assistant_message]);
        // Reload conversation list to pick up auto-generated title
        const updatedList = await api.listConversations();
        setConversations(updatedList);
      } finally {
        setLoading(false);
      }
    },
    [activeId],
  );

  return {
    conversations,
    activeId,
    messages,
    loading,
    selectConversation,
    createConversation,
    deleteConversation,
    renameConversation,
    sendMessage,
  };
}
