import { QueryClient } from '@tanstack/react-query';
import { isApiError } from '@/api/client';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5_000,
      refetchOnWindowFocus: true,
      retry: (count, error) => !(isApiError(error) && error.status < 500) && count < 2,
    },
    mutations: { retry: false },
  },
});
